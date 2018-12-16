# Reinforcement Learning 2048

An AI BOT playing game 2048 by using reinforcement learning

## Overview

### Demo

### The Elements of 2048 Reinforcement Learning problem

- Objective: Get the highest score / max tile. i.e. Live as long as it can while maintaining good board state.

* State: An 4x4 grid with numbers of tiles in value of power of 2.
* Action: Shift board UP, DOWN, LEFT, RIGHT
* Reward: Increment of score or score with other metrics.

### Result

## Usage

Dependencies

* `tensorflow`
* `numpy`
* `pyyaml`

### Basic Game Play

```txt
$ python3 RL2048/Game/Play.py
Play mode:
1. Keyboard (use w, a, s, d, exit with ^C or ^D)
2. Random

 select:
```

* Keyboard mode
* Random mode

### Training model

```txt
$ python3 RL2048/Learning/backward.py
```

* TRAIN_MODE.NORMAL: Normal training process
    * Use only NN itself
* TRAIN_MODE.WITH_RANDOM
    * With a little chance to move randomly

### Default file locations

* Model (ckpt): `./model`
* Last game status: `training_game.yaml`
* Training log: `training.log`

> If you have trouble that can't find RL2048 module. (`ModuleNotFoundError: No module named 'RL2048'`)
>
> You sould make sure your workspace is in the main directory of this project. Then execute code like this.

```sh
export PYTHONPATH=$PYTHONPATH:/path/to/this/project/ReinforcementLearning2048; python3 RL2048/Learning/backward.py
```

> Or add the following lines to every top of the codes.

```py
import sys
sys.path.append('/path/to/this/project/ReinforcementLearning2048')
```

## Heuristic

Artificial Intelligence: How many artifact, how many intelligence!

### Traditonal Tree-search algorithm

The [Monte Carlo tree search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) algorithm

#### Minimax search with alpha-beta pruning

* Monotonicity
* Smoothness
* Free Tiles

### Q-Learning in Deep Learning - Deep Q Network (DQN)

### Policy Gradient in Deep Learning - Deep Deterministic Policy Gradient (DDPG)

## Problems

* Network is too stupid that it keep taking invalid aciton
* Loss become too small and it seems that Network learned nothing in the first 100 round.

## Notes

* [Reinforcement Learning Notes](https://github.com/daviddwlee84/DeepLearningPractice/blob/master/Notes/Technique/Reinforcement_Learning.md)
* There is a more elegant way to store a class object in yaml format by defining it as a subclass of yaml.YAMLObject. ([PyYAML Documentation](https://pyyaml.org/wiki/PyYAMLDocumentation) - `Constructors, representers, resolvers` section)

## Links

### Similar Project

Use Machine Learning

* [tjwei/2048-NN](https://github.com/tjwei/2048-NN) - Max tile 16384, 94% win rate
* [georgwiese/2048-rl](https://github.com/georgwiese/2048-rl)
    * [slides](https://docs.google.com/presentation/d/1I9RS3SMdMp8Uk9C6eyS6jK_w_34BKCrvkN-kWau1MU4/edit?usp=sharing)
* [nneonneo/2048-ai](https://github.com/nneonneo/2048-ai)
* [navjindervirdee/2048-deep-reinforcement-learning](https://github.com/navjindervirdee/2048-deep-reinforcement-learning) - Max tile 4096, 10% win rate

Use Traditional AI

* [daviddwlee84/2048-AI-BOT](https://github.com/daviddwlee84/2048-AI-BOT) - This was me and my friend Tom attending AI competition in 2014.
* [ovolve/2048-AI](https://github.com/ovolve/2048-AI) - 90% win rate
    * [demo](https://ovolve.github.io/2048-AI/)

Simple Game Play

* Python
   * [yangshun/2048-python](https://github.com/yangshun/2048-python)
   * [luliyucoordinate/Python2048](https://github.com/luliyucoordinate/Python2048)
* JavaScript
   * [gabrielecirulli/2048](https://github.com/gabrielecirulli/2048) - almost 10k stars
      * [demo](https://play2048.co/)
   * [GetMIT](https://mitchgu.github.io/GetMIT/)

### Article and Paper

* [Stackoverflow - What is the optimal algorithm for the game 2048?](https://stackoverflow.com/questions/22342854/what-is-the-optimal-algorithm-for-the-game-2048/)
* [MIT - Deep Reinforcement Learning for 2048](http://www.mit.edu/~amarj/files/2048.pdf)
* [Reddit - TDL, N-Tuple Network](https://www.reddit.com/r/2048/comments/2s6m8o/2048_ai_that_has_a_97_win_rate_tdl_ntuple_network/) - 97% win rate
    * [paper](http://www.cs.put.poznan.pl/mszubert/pub/szubert2014cig.pdf)
    * [demo](https://solver2048.appspot.com/#)
* [Stanford - AI Plays 2048](http://cs229.stanford.edu/proj2016/report/NieHouAn-AIPlays2048-report.pdf)

### Others

* [Key listener in Python](https://stackoverflow.com/questions/11918999/key-listeners-in-python)
