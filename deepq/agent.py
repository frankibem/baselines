import numpy as np
from cntk import Value
from cntk.initializer import he_uniform
from cntk.layers import Sequential, Convolution2D, Dense, default_options
from cntk.layers.typing import Tensor
from cntk.learners import adam, learning_rate_schedule, momentum_schedule, UnitType
from cntk.logging import TensorBoardProgressWriter
from cntk.ops import abs, element_select, relu, reduce_sum, square, input_variable, \
    argmax, reduce_mean, one_hot, minus, less, stop_gradient
from cntk.ops.functions import CloneMethod
from cntk.train import Trainer


class LearningAgent(object):
    def __init__(self, state_dim, action_dim, gamma=0.99, learning_rate=1e-4, momentum=0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        with default_options(activation=relu, init=he_uniform()):
            # Convolution filter counts were halved to save on memory, no gpu :(
            self.model = Sequential([
                Convolution2D((8, 8), 16, strides=4, name='conv1'),
                Convolution2D((4, 4), 32, strides=2, name='conv2'),
                Convolution2D((3, 3), 32, strides=1, name='conv3'),
                Dense(256, init=he_uniform(scale=0.01), name='dense1'),
                Dense(action_dim, activation=None, init=he_uniform(scale=0.01), name='actions')
            ])
            self.model.update_signature(Tensor[state_dim])

        # Create the target model as a copy of the online model
        self.target_model = None
        self.update_target()

        self.pre_states = input_variable(state_dim, name='pre_states')
        self.actions = input_variable(action_dim, name='actions')
        self.post_states = input_variable(state_dim, name='post_states')
        self.rewards = input_variable((), name='rewards')
        self.terminals = input_variable((), name='terminals')
        self.is_weights = input_variable((), name='is_weights')

        predicted_q = reduce_sum(self.model(self.pre_states) * self.actions, axis=0)

        # DQN - calculate target q values
        # post_q = reduce_max(self.target_model(self.post_states), axis=0)

        # DDQN - calculate target q values
        online_selection = one_hot(argmax(self.model(self.post_states), axis=0), self.action_dim)
        post_q = reduce_sum(self.target_model(self.post_states) * online_selection, axis=0)

        post_q = (1.0 - self.terminals) * post_q
        target_q = stop_gradient(self.rewards + self.gamma * post_q)

        # Huber loss
        delta = 1.0
        self.td_error = minus(predicted_q, target_q, name='td_error')
        abs_error = abs(self.td_error)
        errors = element_select(less(abs_error, delta), square(self.td_error) * 0.5, delta * (abs_error - 0.5 * delta))
        loss = reduce_mean(self.is_weights * errors, name='loss')  # weighted error

        # Adam based SGD
        lr_schedule = learning_rate_schedule(learning_rate, UnitType.minibatch)
        m_scheule = momentum_schedule(momentum)
        vm_schedule = momentum_schedule(0.999)

        self._learner = adam(self.model.parameters, lr_schedule, m_scheule, variance_momentum=vm_schedule)
        self.writer = TensorBoardProgressWriter(log_dir='metrics', model=self.model)
        self.trainer = Trainer(self.model, (loss, None), [self._learner], self.writer)

    def act(self, state, epsilon):
        """
        Selects an action to take based on the epsilon greedy method
        :param state: The current state
        :param epsilon: Determines the amount of exploration. (1 - full exploration, 0 - no exploration)
        """
        if np.random.randn(1) < epsilon:
            # Explore (random action)
            return np.random.choice(self.action_dim)
        else:
            # Exploit (greedy action based on knowledge)
            return self.model.eval(state).argmax()

    def train(self, s, a, r, s_, t, w):
        """
        Updates the network weights using the given minibatch data
        :param s: Tensor[state_dim] Current state
        :param a: Tensor[int] Action taken at state s
        :param r: Tensor[float] State resulting from taking action a at state s
        :param s_: Tensor[state_dim] Reward received for taking action a at state s
        :param t: Tensor[boolean] True if s_ was a terminal state and false otherwise
        :param w: Tensor[float] Importance sampling weights
        """
        a = Value.one_hot(a.tolist(), self.action_dim)
        td_error = self.trainer.train_minibatch({
            self.pre_states: s,
            self.actions: a,
            self.rewards: r,
            self.post_states: s_,
            self.terminals: t,
            self.is_weights: w
        }, outputs=[self.td_error])
        return td_error[0]

    def update_target(self):
        """
        Update the target network using the online network weights
        """
        self.target_model = self.model.clone(CloneMethod.freeze)

    def checkpoint(self, filename):
        self.trainer.save_checkpoint(filename)

    def save_model(self, filename):
        self.model.save(filename)
