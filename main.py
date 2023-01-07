import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import TD3, PPO, A2C, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common import results_plotter
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.results_plotter import plot_results
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

from env.portfolio_trading_env import PortfolioTradingEnv as Env

def make_env(rank):
    """
    Utility function for multiprocessed env.

    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = Env(portfolio_stocks=["AAPL", "AMZN", "NFLX", "GOOGL"], initial_cash=100000, start_day="2012-01-01", 
                  end_day="2016-12-30")
        return env
    
    return _init

if __name__ == '__main__':
    prs = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                  description="""Args for Env""")

    num_cpu = 8  # Number of processes to use
    env = Env(portfolio_stocks=["AAPL", "AMZN", "NFLX", "GOOGL"], initial_cash=100000, start_day="2012-01-01", 
                  end_day="2016-12-30")
    
    if args.method == 'randomm': #baseline
        env.reset()
        for i in range(100000):
            action = env.action_space.sample()
            obs, reward, done, _ = env.step(action)
            if done:
                env.reset()
    else:
        # Create the vectorized environment
        env = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)

        if args.method == 'ppo':
            model = PPO('MlpPolicy', env, gamma=0.95, verbose=1)
            model.learn(total_timesteps=800000) 
            
        elif args.method == 'a2c':
            
            model = A2C('MlpPolicy', env, gamma=0.95, verbose=0)
            
            model.learn(total_timesteps=800000)
        elif args.method == "sac":
            model = SAC('MlpPolicy', env, gamma=0.95, verbose=0)
            
            model.learn(total_timesteps=800000)
        else:
            print('Invalid Algorithm')
