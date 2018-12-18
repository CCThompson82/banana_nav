# Udacity DRLND Project: Navigation

## Project Details
### Summary
The objective of this project is to implement a solution to a simplified version of the Unity 
Banana Collector problem, from Unity ML-Agents [[video](https://www.youtube.com/watch?v=heVMs3t9qSk&feature=youtu.be)]. 
Whereas the version shown above is a multi-agent game, the task for this project is simply to train 
an agent capable of collecting yellow bananas while avoiding purple.  
### Task Specifics
* The state space is a 37-feature ray-cast perception for local objects.  
* The action space movement in the 4 cardinal directions.  
* The objective is to collect as many yellow bananas as possible (+1), while avoiding purple bananas (-1). 
* The task is considered solved for this project if an average episode score of 13 is achieved over
the course of 100 episodes.  


## Installation
In order to maintain compatibility across different operating systems, and to contain the various 
non-python dependencies for the udacity deep reinforcement learning nanodegree, this repository 
is built off of a base docker image available freely from my dockerhub repository.  The 
instructions for downloading the 2 essential dependencies are hyperlinked in the `Requirements` 
section below.  Instructions for setting up and running the docker environment are described in the 
`Setup Environment` section.    

### Requirements 
* [docker](https://docs.docker.com/install/)  
    - find and follow the instructions for your operating system
* [cmake](https://cmake.org/install/)

### Setup Environment

1. Modify the cmake CONFIG defaults to match your preferences.  Particularly, the directory in 
which data should be stored should be updated to match your system preferences.   
2. Setup the development environment in a Docker container with the following command:
    - `make init`
    
    This command gets the resources for training and testing, and then prepares the Docker image for the experiments.
3. After creating the Docker image, run the following command.

- `make create-container`

    The above command creates a Docker container from the Docker image which we create with `make init`, and then
login to the Docker container.  Only the first time you need to create a Docker container, from the image created in `make init` command.
`make create-container` creates and launch the banana_nav container.
After creating the container, you just need run `make start-container`.

## Train the product model agent from scratch 
1. start the container:
    * `make start-container` or `make create-container`
2. update the `experiment_id` parameter in `config/model.json` to a unique namespace.
3. run the following command from the docker container: 
    * `python3 ./scripts/train.py`

Voila! A tdqm progress bar should provide the necessary information to understand agent performance 
over the course of training.  A separate evaluation script can be run after updating the 
relevant fields in the `config/eval.json` file, and then running 
`./scripts/evaluate_model_checkpoint.py`.      


## Credits

This repository was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [cookiecutter-docker-science](https://docker-science.github.io/) project template.
