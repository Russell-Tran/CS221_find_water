# Stanford CS 221 (Principles of Artificial Intelligence) Final Project: Finding Water in Minecraft  
[Lydia Chan](https://github.com/LydiaChan528), [Russell Tran](https://github.com/Russell-Tran)  
13 December 2019  

We use Deep Q-Learning (DQN) to train an agent to search for bodies of water in the video game Minecraft. The agent reads in raw pixel inputs and has the controls of a normal player.   

## Directory 
* `code`: Run the training algorithms and simulate Minecraft. To run, call `python3` {`find_water_baseline.py`, `find_water_dqn0.py`, `find_water_q_learning.py`}
* `environments`: These are the xml files which represent different Minecraft worlds/environments in which the agent can roam. These xml files are parsed by Minecraft Malmo--refer to their documentation for the formatting. The MineRL platform is capable of taking these Minecraft Malmo environments (which Malmo calls "missions") and using them as OpenAI gym environmetns.
* `out`: Data output on our runs
* `poster data`: Assets for our poster

## Below are the necessary dependencies for macOS and Windows:

tensorflow==1.14

minerl==0.2.9

pandas==0.24.8

gym==0.15.3

mujoco-py>=2.0.2.8

mpi4py==3.0.3

baselines==0.1.5

lxml==4.4.1

psutil>=5.6.2
