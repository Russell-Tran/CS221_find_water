"""
FIND WATER BASELINE
Lydia Chan, Russell Tran
22 November 2019

This script runs the baseline to our Find Water task.
In our baseline, our agent completely ignores all sensory input (observations),
and simply solves the task by brute force moving in a snakelike fashion until
inevitably hitting the target.
"""

# ========================
# Import global modules
# ========================
import os
import pandas as pd
from datetime import datetime
import gym
import itertools
import minerl
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers
import baselines.common.tf_util as U
from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer
from baselines.common.schedules import LinearSchedule
import logging
import coloredlogs
coloredlogs.install(logging.INFO)

# =========================
# Import local modules
# =========================
import register_custom_environment
import util

# =========================
# Define CONFIG CONSTANTS
# =========================
ALGORITHM_NAME = "baseline"
ENVIRONMENT_NAME = "boxed_water_medium"
NUM_GAMES_TO_PLAY = 200
MINECRAFT_MISSION_XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "{}.xml".format(ENVIRONMENT_NAME))
ENVIRONMENT_ID = 'russell-water-v0'
VERBOSE = True
BASELINE_STEPS_FORWARD = 150
BASELINE_TURNS_TO_NINETY_DEG = 9
BASELINES_RESTS_PER_PIVOT = 5


# =========================
# Print loading message
# =========================
print("MINECRAFT_MISSION_XML_PATH = {}".format(MINECRAFT_MISSION_XML_PATH))
print('=' * 60)
print("Find Water: Minecraft is setting itself up. Please be patient...")
print('=' * 60)

# =========================
# Register our custom environment
# This step is necessary to call gym.make() later
# =========================
register_custom_environment.register(environment_id=ENVIRONMENT_ID, xml_path=MINECRAFT_MISSION_XML_PATH)



def run():
    # Keep track of data
    datasaver = util.DataSaver(ALGORITHM_NAME, ENVIRONMENT_NAME)

    # Create the environment
    env = gym.make(ENVIRONMENT_ID)

    # Keep track of the rewards for each episode
    episode_rewards = [0.0]

    # Reset the gym environment's and return the initial observation
    obs = (env.reset())

    """ 
    Run the games.
    Each iteration t represents a step taken
    by the agent. Loop indefinitely until the number of games
    completed exceeds NUM_GAMES_TO_PLAY
    """
    game_stats = []
    games_completed = 0
    should_break = False

    # The total number of steps taken
    t = 0
    old_t = 0 

    def take_environment_step(action):
        """
        Takes one step in the environment
        Returns a boolean |shouldRestartPolicyCycle|, which is 
        True if reached terminal state or we hit the limit of games completed;
        False otherwise
        """
        nonlocal game_stats
        nonlocal games_completed
        nonlocal should_break
        nonlocal episode_rewards
        nonlocal t
        nonlocal old_t

        # Check before continuing
        if games_completed >= NUM_GAMES_TO_PLAY:
            print("Hit the limit of games completed. Breaking")
            should_break = True
            return True
        if VERBOSE:
            if t % 1000 == 0:
                print("t = {}".format(t))

        # Increment the time step
        t += 1
        # Take the actual step in the environment
        new_obs, rew, done, _ = env.step(action)
        # Update the rewards for this episode
        episode_rewards[-1] += rew

        # Determine whether to render the environment
        is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
        if is_solved:
            # Show off the result
            env.render()

        # Did we reach a terminal state?
        if done:
            obs = env.reset()
            episode_rewards.append(0)

            # Print game stats for this completed game
            if len(episode_rewards) % 1 == 0:
                print("Game #{} was just completed".format(games_completed))
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.dump_tabular()
                games_completed += 1
                game_stats.append({
                    "steps" : t,
                    "episodes" : len(episode_rewards),
                    "mean episode reward" : round(np.mean(episode_rewards[-101:-1]), 1),
                    "reward_for_this_episode" : episode_rewards[-2],
                    "delta_t" : t - old_t
                    })
                old_t = t
                datasaver.save_list_of_dicts(game_stats)

            return True

        else:
            return False


    def policy_forward(quantity):
        """Walk forward |quantity| times
        """
        for i in range(quantity):
            action = {'forward' : 1}
            shouldRestartPolicyCycle = take_environment_step(action)
            if shouldRestartPolicyCycle:
                return shouldRestartPolicyCycle
        return False

    def policy_counterclockwise(quantity):
        """Rotate counterclockwise |quantity| times
        """
        for i in range(quantity):
            action = {'camera' : [0, -10]}
            shouldRestartPolicyCycle = take_environment_step(action)
            if shouldRestartPolicyCycle:
                return shouldRestartPolicyCycle
        return False

    def policy_clockwise(quantity):
        """Rotate clockwise |quantity| times
        """
        for i in range(quantity):
            action = {'camera' : [0, 10]}
            shouldRestartPolicyCycle = take_environment_step(action)
            if shouldRestartPolicyCycle:
                return shouldRestartPolicyCycle
        return False

    def policy_rest(quantity):
        """ Rest for |quantity| times
        """
        for i in range(quantity):
            action = {}
            shouldRestartPolicyCycle = take_environment_step(action)
            if shouldRestartPolicyCycle:
                return shouldRestartPolicyCycle
        return False

    """
    LOOP THROUGH OUR BASELINE POLICY, which ignores all observations.
    """
    for policy_cycles in itertools.count():
        # Check whether to stop altogether
        if should_break:
            break
        # ==========================
        # === Walk straight down ===
        # First go forward BASELINE_STEPS_FORWARD number of times
        shouldRestartPolicyCycle = policy_forward(BASELINE_STEPS_FORWARD)
        if shouldRestartPolicyCycle:
            continue
        # ==========================
        # Rest before turning to slow velocity
        shouldRestartPolicyCycle = policy_rest(BASELINES_RESTS_PER_PIVOT)
        if shouldRestartPolicyCycle:
            continue
        # ==========================
        # === Pivot after hitting the wall ===
        # Then turn counterclockwise by 90 degrees
        shouldRestartPolicyCycle = policy_counterclockwise(BASELINE_TURNS_TO_NINETY_DEG)
        if shouldRestartPolicyCycle:
            continue
        # Then go forward 1 time
        shouldRestartPolicyCycle = policy_forward(1)
        if shouldRestartPolicyCycle:
            continue
        # ==========================
        # Rest before turning to slow velocity
        shouldRestartPolicyCycle = policy_rest(BASELINES_RESTS_PER_PIVOT)
        if shouldRestartPolicyCycle:
            continue
        # Then turn counterclockwise by 90 degrees
        shouldRestartPolicyCycle = policy_counterclockwise(BASELINE_TURNS_TO_NINETY_DEG)
        if shouldRestartPolicyCycle:
            continue
        # ==========================
        # === Walk back the way we came, except one column down now ===
        # Now go forward BASELINE_STEPS_FORWARD number of times
        shouldRestartPolicyCycle = policy_forward(BASELINE_STEPS_FORWARD)
        if shouldRestartPolicyCycle:
            continue
        # ==========================
        # ==========================
        # Rest before turning to slow velocity
        shouldRestartPolicyCycle = policy_rest(BASELINES_RESTS_PER_PIVOT)
        if shouldRestartPolicyCycle:
            continue
        # === Pivot again ===
        # Then turn clockwise by 90 degrees
        shouldRestartPolicyCycle = policy_clockwise(BASELINE_TURNS_TO_NINETY_DEG)
        if shouldRestartPolicyCycle:
            continue
        # Then go forward 1 time
        shouldRestartPolicyCycle = policy_forward(1)
        if shouldRestartPolicyCycle:
            continue
        # ==========================
        # Rest before turning to slow velocity
        shouldRestartPolicyCycle = policy_rest(BASELINES_RESTS_PER_PIVOT)
        if shouldRestartPolicyCycle:
            continue
        # Then turn clockwise by 90 degrees
        shouldRestartPolicyCycle = policy_clockwise(BASELINE_TURNS_TO_NINETY_DEG)
        if shouldRestartPolicyCycle:
            continue
        # === THE CYCLE IS COMPLETE ===

if __name__ == '__main__':
    # Set up a session with at most 8 CPUs used
    with U.make_session(8):
        run()
        