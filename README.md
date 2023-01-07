# gym-portfoliotrading

## Introduction
This repository contains an OpenAI Gym Environment for portfolio trading with data from Google, Amazon, APPLE, and NetFlex. The complete environment code and indicators can be found in the env directory. The stable baseline 3 has been used for RL agent training.

## Requirements

-   Python 3.7
-   Pytorch 1.8.1
-   OpenAI Gym 0.22.0
-   Matplotlib
-   Stable-baseline3
-   Pandas
-   Numpy

## To run the code you need to run this command:

```
python main.py -method ppo
```
This will run the PPO. You can any algorithm by just specifying the name of the algorithm.
