"""
find_water.py
Lydia Chan, Russell Tran
22 November 2019

Run this script to perform the training.
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
import basic_q_learning
import util

# =========================
# Define CONFIG CONSTANTS
# =========================
ALGORITHM_NAME = "q_learning"
ENVIRONMENT_NAME = "boxed_water_medium"
NUM_GAMES_TO_PLAY = 200
MINECRAFT_MISSION_XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "{}.xml".format(ENVIRONMENT_NAME))
ENVIRONMENT_ID = 'russell-water-v0'
VERBOSE = True

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


# Set up the Q-learning algorithm

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a

def action_function(obs):
    # These correspond to the enumerated actions from the function action_wrapper() below
    return [0, 1, 2, 3]


# Return a single-element list containing a binary (indicator) feature
# for the existence of the (state, action) pair.  Provides no generalization.
def identityFeatureExtractor(state, action):
    featureKey = (state, action)
    featureValue = 1
    return [(featureKey, featureValue)]


q_learning = basic_q_learning.QLearningAlgorithm(
    actions=action_function, 
    discount=1, 
    featureExtractor=identityFeatureExtractor, 
    explorationProb=0.3
    )


def action_wrapper(action_int):
    """
    Convert the enumerated action int (0 through 3, inclusive)
    provided by our agent into a more meaningful action that
    can be interpreted as an action in the Minecraft environment
    """
    # Action space
    act = {}

    # Convert the enumeration
    if action_int == 0:
        # Involve a jump in this next step
        act['jump'] = 1
    elif action_int  == 1:
        # Turn right
        act['camera'] = [0, 30]
    elif action_int  == 2:
        # Turn left
        act['camera'] = [0, -30]
    elif action_int == 3:
        # Go forward
        act['forward'] = 1

    return act.copy()

def observation_wrapper(obs):
    """ Reformat the observations in a convenient way.
    Converts the pixel values from the usual [0, 255] value range
    to a range of [-0.5, +0.5].
    DOES NOT incorporate the compass angle at all.
    """
    # Convert pixel values
    pov = obs['pov'].astype(np.float32)/255.0 - 0.5
    return pov 

if __name__ == '__main__':
    # Set up a session with at most 8 CPUs used
    with U.make_session(8):
        # Keep track of data
        datasaver = util.DataSaver(ALGORITHM_NAME, ENVIRONMENT_NAME)

        # Create the environment
        env = gym.make(ENVIRONMENT_ID)

        # Get dimensions of POV observation space
        # where env.observation_space.spaces =
        # OrderedDict([('compassAngle', Box()), ('inventory', Dict(dirt:Box())), ('pov', Box(64, 64, 3))])
        # For 'pov', the first two dimensions are x & y coordinates of pixels and the third dimension is
        # for each RGB
        spaces = env.observation_space.spaces['pov']
        shape = list(spaces.shape)
        shape[-1] += 1 # effectively change shape (64, 64, 3) --> (64, 64, 4)


        # Keep track of the rewards for each episode
        episode_rewards = [0.0]

        # Reset the gym environment's and return the initial observation
        obs = (env.reset())
        obs = observation_wrapper(obs)


        """ 
        Run the games.
        Each iteration in this loop represents a step taken
        by the agent. Loop indefinitely until the number of games
        completed exceeds NUM_GAMES_TO_PLAY
        """
        game_stats = []
        games_completed = 0
        old_t = 0
        for t in itertools.count():
            if games_completed >= NUM_GAMES_TO_PLAY:
                print("Hit the limit of games completed. Breaking")
                break
            if VERBOSE:
                if t % 1000 == 0:
                    print("t = {}".format(t))

            # Take action 
            # Since the pixel displays represent our states
            # then our states are synonymous with observations ("obs" and "new_obs")
            action = q_learning.getAction(totuple(obs))
            new_obs, rew, done, _ = env.step(action_wrapper(action))
            new_obs = observation_wrapper(new_obs)

            # Update the rewards for this episode
            reward = rew
            episode_rewards[-1] += rew

            # Incorporate the feedback depending on whether we are done with this episode
            if done:
                # Mark the new state as None to indicate terminal state
                q_learning.incorporateFeedback(state=totuple(obs), action=action, reward=reward, newState=None)
                obs = env.reset()
                obs = observation_wrapper(obs)
                episode_rewards.append(0)
            else:
                # The usual incorporation of feedback for non-terminal states
                q_learning.incorporateFeedback(state=totuple(obs), action=action, reward=reward, newState=totuple(new_obs))
                # Update current state
                obs = new_obs

            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            if is_solved:
                # Show off the result
                env.render()

            # Print game stats for this completed game
            if done and len(episode_rewards) % 1 == 0:
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