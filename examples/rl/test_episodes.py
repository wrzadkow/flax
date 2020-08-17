import gym
import time
import flax
import itertools

from env import create_env
from remote import get_state
from agent import greedy_action

def test(n_episodes: int, model: flax.nn.base.Model, render: bool = False):
    test_env = create_env()
    if render:
        test_env = gym.wrappers.Monitor(
            test_env, "./rendered/" + "ddqn_pong_recording", force=True
        )
    for e in range(n_episodes):
        obs = test_env.reset()
        state = get_state(obs)
        total_reward = 0.0
        for t in itertools.count():
            action, _ = greedy_action(model, state)
            action = action[0]
            obs, reward, done, info = test_env.step(action)
            total_reward += reward
            if render:
                test_env.render()
                time.sleep(0.01)
            if not done:
                next_state = get_state(obs)
            else:
                next_state = None
            state = next_state
            if done:
                print(f"Finished Episode {e} with reward {total_reward}")
                break
    del test_env
    return