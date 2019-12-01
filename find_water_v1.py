"""
FIND_WATER_v1
Lydia Chan, Russell Tran
22 November 2019

THIS IS THE SCRIPT THAT IS USED FOR DQN.
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
ALGORITHM_NAME = "DQN"
ENVIRONMENT_NAME = "boxed_water_medium"
NUM_GAMES_TO_PLAY = 400
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

# Create a q_function ("model") for a DQN algorithm
model = deepq.models.cnn_to_mlp(
    # list of convolutional layers in form of
    # (num_outputs, kernel_size, stride)
    convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
    # list of sizes of hidden layers
    hiddens=[256],
    # set whether we want this to be a dueling DQN
    # see https://github.com/openai/baselines/pull/946/commits/933265b01fb6e17a65a052b62aa1f9d592a67e6f
    dueling=True
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
        act['camera'] = [0, 20]
    elif action_int  == 2:
        # Turn left
        act['camera'] = [0, -20]
    elif action_int == 3:
        # Go forward
        act['forward'] = 1

    return act.copy()

def observation_wrapper(obs):
    """ Reformat the observations in a convenient way.
    Converts the pixel values from the usual [0, 255] value range
    to a range of [-0.5, +0.5].
    Also incorporates the compass angle into all of data points for this given observation.
    """
    # Convert pixel values
    pov = obs['pov'].astype(np.float32)/255.0 - 0.5
    # Get compass angle
    compass = obs['compassAngle']
    compass_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32)*compass
    compass_channel /= 180.0
    # Incorporate 
    return np.concatenate([pov, compass_channel], axis=-1)

if __name__ == '__main__':
    # Set up a session with at most 8 CPUs used
    with U.make_session(8):
        # Keep track of data
        datasaver = util.DataSaver(ALGORITHM_NAME, ENVIRONMENT_NAME)

        # Create the environment
        env = gym.make(ENVIRONMENT_ID)

        # somehow, trying to render it destroys this
        # env.render()

        # Get dimensions of POV observation space
        # where env.observation_space.spaces =
        # OrderedDict([('compassAngle', Box()), ('inventory', Dict(dirt:Box())), ('pov', Box(64, 64, 3))])
        # For 'pov', the first two dimensions are x & y coordinates of pixels and the third dimension is
        # for each RGB
        spaces = env.observation_space.spaces['pov']
        shape = list(spaces.shape)
        shape[-1] += 1 # effectively change shape (64, 64, 3) --> (64, 64, 4)

        # Create all the functions necessary to train the model
        # https://github.com/openai/baselines/blob/master/baselines/deepq/build_graph.py
        # Descriptions for the following functions:
        # act : select an action given observation
        # train : optimize the error in Bellman's equation
        # update_target : copy the parameters from the optimized
        # Q function to the target Q function
        # debug : a dictionary of functions that print debug data
        # such as q_values
        act, train, update_target, debug = deepq.build_train(
            # make_obs_ph is a function that takes a name and
            # creates a placeholder of input with that name
            # and BatchInput creates a placeholder for a batch of tensors of a given shape and dtype
            make_obs_ph=lambda name: deepq.utils.BatchInput(shape
                , name=name),
            q_func=model,
            # We have a total of 4 different actions, and those actions are enumerated 0 through 3, inclusive
            num_actions=4,
            # Adam is a variation of Stochastic Gradient Descent
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )

        # Create the replay buffer
        # with a limit of 30,000 transitions stored in the buffer
        replay_buffer = ReplayBuffer(30000)
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=450000, initial_p=1.0, final_p=0.02)

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

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

            # Take action and update exploration to the newest value
            action = act(obs[None], update_eps=exploration.value(t))[0]
            new_obs, rew, done, _ = env.step(action_wrapper(action))
            new_obs = observation_wrapper(new_obs)

            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs

            episode_rewards[-1] += rew
            if done:
                obs = env.reset()
                obs = observation_wrapper(obs)
                episode_rewards.append(0)

            is_solved = t > 100 and np.mean(episode_rewards[-101:-1]) >= 200
            if is_solved:
                # Show off the result
                env.render()
            else:
                # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
                if t > 1000:
                    obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(32)
                    train(obses_t, actions, rewards, obses_tp1, dones, np.ones_like(rewards))
                # Update target network periodically.
                if t % 1000 == 0:
                    update_target()

            # Print game stats for this completed game
            if done and len(episode_rewards) % 1 == 0:
                print("Game #{} was just completed".format(games_completed))
                logger.record_tabular("steps", t)
                logger.record_tabular("episodes", len(episode_rewards))
                logger.record_tabular("mean episode reward", round(np.mean(episode_rewards[-101:-1]), 1))
                logger.record_tabular("% time spent exploring", int(100 * exploration.value(t)))
                logger.dump_tabular()
                games_completed += 1
                game_stats.append({
                    "steps" : t,
                    "episodes" : len(episode_rewards),
                    "mean episode reward" : round(np.mean(episode_rewards[-101:-1]), 1),
                    "% time spent exploring" : int(100 * exploration.value(t)),
                    "reward_for_this_episode" : episode_rewards[-2],
                    "delta_t" : t - old_t
                    })
                old_t = t
                datasaver.save_list_of_dicts(game_stats)