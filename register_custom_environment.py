# =========================
""" Do the work necessary to register our
custom environment into the gym.
This code came from minerl/env/__init__.py
at https://github.com/minerllabs/minerl
"""
# Perform the registration.
import minerl
import gym
#from gym.envs.registration import register
from collections import OrderedDict
from minerl.env import spaces
from minerl.env.core import MineRLEnv, missions_dir
import numpy as np

def register(environment_id, xml_path):
    """
    Given an xml_path and a desired environment_id,
    registers the Minecraft mission as a gym environment.

    The environment_id is a string which you "register" here
    and then call later when doing gym.make(environment_id).

    The xml_path is the path to the Malmo mission file, which is
    in xml file format, of course.
    """
    ENVIRONMENT_ID = environment_id
    XML_PATH = xml_path

    def make_navigate_text(top, dense):
        """
        Helper function to construct the description for this gym environment.
        TODO: Copied as an example and needs to be removed / changed
        """
        navigate_text = """
            .. image:: ../assets/navigate{}1.mp4.gif
                :scale: 100 %
                :alt: 

            .. image:: ../assets/navigate{}2.mp4.gif
                :scale: 100 %
                :alt: 

            .. image:: ../assets/navigate{}3.mp4.gif
                :scale: 100 %
                :alt: 

            .. image:: ../assets/navigate{}4.mp4.gif
                :scale: 100 %
                :alt: 

            In this task, the agent must move to a goal location denoted by a diamond block. This represents a basic primitive used in many tasks throughout Minecraft. In addition to standard observations, the agent has access to a “compass” observation, which points near the goal location, 64 meters from the start location. The goal has a small random horizontal offset from the compass location and may be slightly below surface level. On the goal location is a unique block, so the agent must find the final goal by searching based on local visual features.

            The agent is given a sparse reward (+100 upon reaching the goal, at which point the episode terminates). 
        """
        if dense:
            navigate_text += "**This variant of the environment is dense reward-shaped where the agent is given a reward every tick for how much closer (or negative reward for farther) the agent gets to the target.**\n"
        else: 
            navigate_text += "**This variant of the environment is sparse.**\n"

        if top is "normal":
            navigate_text += "\nIn this environment, the agent spawns on a random survival map.\n"
            navigate_text = navigate_text.format(*["" for _ in range(4)])
        else:
            navigate_text += "\nIn this environment, the agent spawns in an extreme hills biome.\n"
            navigate_text = navigate_text.format(*["extreme" for _ in range(4)])
        return navigate_text

    # TODO: Figure out what this represents
    navigate_action_space = spaces.Dict({
        "forward": spaces.Discrete(2),
        "back": spaces.Discrete(2),
        "left": spaces.Discrete(2),
        "right": spaces.Discrete(2),
        "jump": spaces.Discrete(2),
        "sneak": spaces.Discrete(2),
        "sprint": spaces.Discrete(2),
        "attack": spaces.Discrete(2),
        "camera": spaces.Box(low=-180, high=180, shape=(2,), dtype=np.float32),
        "place": spaces.Enum('none', 'dirt')})

    # TODO: Figure out what this represents
    navigate_observation_space = spaces.Dict({
        'pov': spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
        'inventory': spaces.Dict(spaces={
            'dirt': spaces.Box(low=0, high=2304, shape=(), dtype=np.int)
        }),
        'compassAngle': spaces.Box(low=-180.0, high=180.0, shape=(), dtype=np.float32)
    })

    # This is the function call where we actually register
    # the environment
    gym.envs.registration.register(
        id=ENVIRONMENT_ID,
        entry_point='minerl.env:MineRLEnv',
        kwargs={
            'xml': XML_PATH,
            'observation_space': navigate_observation_space,
            'action_space': navigate_action_space,
            'docstr': make_navigate_text('normal', False)
        },
        max_episode_steps=6000,
    )