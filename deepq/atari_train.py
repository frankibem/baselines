import gym
import numpy as np

from common.atari_wrappers import EpisodicLifeEnv, MaxAndSkipEnv, WarpFrame, ClipRewardEnv, FrameStack, NormalizedEnv
from common.schedules import LinearSchedule
from deepq.agent import LearningAgent
from deepq.replay_buffer import PrioritizedReplayBuffer


def wrap_env(env):
    """
    Wraps the Gym environment such that the end of a life is considered the end
    of an episode (EpisodicLifeEnv), every 4th frame is returned (MaxAndSkipFrame),
    the frames are grayscaled and resized to 84x84 (WarpFrame), pixels are normalized,
    frames are buffered and stacked and rewards are clipped to [-1, 1]
    """
    env = EpisodicLifeEnv(env)
    env = MaxAndSkipEnv(env, skip=4)
    env = WarpFrame(env)
    env = NormalizedEnv(env)
    env = FrameStack(env, 4)
    env = ClipRewardEnv(env)
    return env


def main(env_name,
         train_freq=1,
         target_update_freq=1000,
         batch_size=32,
         train_after=64,
         final_gamma=0.02,
         max_timesteps=2000000,
         buffer_size=10000,
         prioritized_replay_alpha=0.6,
         prioritized_replay_beta=0.4,
         prioritized_replay_eps=1e-6,
         log_freq=1,
         checkpoint_freq=10000):
    env = gym.make(env_name)
    env = wrap_env(env)

    state_dim = (4, 84, 84)
    action_dim = env.action_space.n

    agent = LearningAgent(state_dim, action_dim)
    logger = agent.writer

    eps_sched = LinearSchedule(1.0, final_gamma, max_timesteps)
    beta_sched = LinearSchedule(prioritized_replay_beta, 1.0, max_timesteps)
    replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha=prioritized_replay_alpha)

    try:
        obs = env.reset()
        episode = 0
        rewards = 0
        steps = 0
        for t in range(max_timesteps):
            # Take action and update exploration to newest value
            action = agent.act(obs, epsilon=eps_sched.value(t))
            obs_, reward, done, _ = env.step(action)

            # Store transition in replay buffer
            replay_buffer.add(obs, action, reward, obs_, float(done))
            obs = obs_

            rewards += reward
            if done:
                steps = t - steps
                episode += 1
                obs = env.reset()

            if t > train_after and (t % train_freq) == 0:
                print('Training...')
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer
                experience = replay_buffer.sample(batch_size, beta=beta_sched.value(t))
                (s, a, r, s_, t, weights, batch_idxes) = experience

                td_errors = agent.train(s, a, r, s_, t, weights)
                new_priorities = np.abs(td_errors) + prioritized_replay_eps
                replay_buffer.update_priorities(batch_idxes, new_priorities)

            if t > train_after and (t % target_update_freq) == 0:
                agent.update_target()

            if done and (episode % log_freq) == 0:
                logger.write_value('rewards', rewards, episode)
                logger.write_value('steps', steps, episode)
                logger.write_value('epsilon', eps_sched.value(t), episode)
                agent.trainer.summarize_training_progress()
                logger.flush()

                rewards = 0
                steps = t

            if t > train_after and (t % checkpoint_freq) == 0:
                agent.checkpoint('model_{}.chkpt'.format(t))
    finally:
        agent.save_model('model.dnn')


if __name__ == '__main__':
    main('KungFuMasterNoFrameskip-v0')
