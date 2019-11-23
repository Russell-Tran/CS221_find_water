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
import baselines.deepq.utils as shit # Changed the import here to fix a bug
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
# =========================
# =========================
# CONFIG CONSTANTS
NUM_GAMES_TO_PLAY = 35
MINECRAFT_MISSION_XML_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "russell_water.xml")
ENVIRONMENT_ID = 'russell-water-v0'
# =========================
# =========================
print("MINECRAFT_MISSION_XML_PATH = {}".format(MINECRAFT_MISSION_XML_PATH))
print("I dunno why but things take a while at this point to launch Minecraft...")
# =========================
# Register our custom environment
# This step is necessary to call upon gym.make() later
register_custom_environment.register(environment_id=ENVIRONMENT_ID, xml_path=MINECRAFT_MISSION_XML_PATH)

# TODO: Figure out what this means
model = deepq.models.cnn_to_mlp(
    convs=[(32, 8, 4), (64, 4, 2), (64, 3, 1)],
    hiddens=[256],
    dueling=True
)

# TODO: Figure out what this means
def action_wrapper(action_int):
    act = {
        "forward": 1,
        "back": 0, 
        "left": 0, 
        "right": 0, 
        "jump": 0, 
        "sneak": 0, 
        "sprint": 1, 
        "attack" : 1, 
        "camera": [0,0],
        "place": 'none'
    }
    if action_int == 0:
        act['jump'] = 1
    elif action_int  == 1:
        act['camera'] = [0, 10]
    elif action_int  == 2:
        act['camera'] = [0, -10]

    return act.copy()

# TODO: Figure out what this means
def observation_wrapper(obs):
    pov = obs['pov'].astype(np.float32)/255.0- 0.5
    compass = obs['compassAngle']

    compass_channel = np.ones(shape=list(pov.shape[:-1]) + [1], dtype=np.float32)*compass
    compass_channel /= 180.0
    
    return np.concatenate([pov, compass_channel], axis=-1)

# TODO: Figure out what this means
if __name__ == '__main__':
    verbose = True
    with U.make_session(8):
        # Create the environment
        env = gym.make(ENVIRONMENT_ID)
        spaces = env.observation_space.spaces['pov']
        shape = list(spaces.shape)
        shape[-1] += 1
        if verbose:
            print("We like made the environment")

        # Create all the functions necessary to train the model
        act, train, update_target, debug = deepq.build_train(
            make_obs_ph=lambda name: shit.BatchInput(shape
                , name=name),
            q_func=model,
            num_actions=4,
            optimizer=tf.train.AdamOptimizer(learning_rate=5e-4),
        )
        if verbose:
            print("We just like made all the functions necessary to train the model")

        # Create the replay buffer
        replay_buffer = ReplayBuffer(30000)
        if verbose:
            print("We just made the replay buffer")
        # Create the schedule for exploration starting from 1 (every action is random) down to
        # 0.02 (98% of actions are selected according to values predicted by the model).
        exploration = LinearSchedule(schedule_timesteps=100000, initial_p=1.0, final_p=0.02)
        if verbose:
            print("We just made the schedule for exploration")

        # Initialize the parameters and copy them to the target network.
        U.initialize()
        update_target()

        episode_rewards = [0.0]
        obs = (env.reset())
        # obs = test_obs
        obs = observation_wrapper(obs)
        

        game_stats = []
        games_completed = 0
        for t in itertools.count():

            if games_completed > NUM_GAMES_TO_PLAY:
                print("Hit the limit of games completed. Breaking")
                break

            

            if verbose:
                if t % 1000 == 0:
                    print("t = {}".format(t))

            # Take action and update exploration to the newest value
            # print(obs[None].shape)
            action = act(obs[None], update_eps=exploration.value(t))[0]
            
            new_obs, rew, done, _ = env.step(action_wrapper(action))
            # new_obs,  rew, done  = test_obs, 1, 0
            new_obs = observation_wrapper(new_obs)
            # Store transition in the replay buffer.
            replay_buffer.add(obs, action, rew, new_obs, float(done))
            obs = new_obs
            # print(new_obs)

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
                    "% time spent exploring" : int(100 * exploration.value(t))

                    })


        # Russell wrote this part
        # Basically it just saves the game stats to a csv
        print("Saving to dataframe")
        df = pd.DataFrame(game_stats)
        now = datetime.now() # current date and time
        date_time_string = now.strftime("%m-%d-%Y-%H%M%S")
        csv_name = "russell_water_v0_run_{}.csv".format(date_time_string)
        print("Saving to csv named {}".format(csv_name))
        df.to_csv(csv_name)
        # End of saving to CSV