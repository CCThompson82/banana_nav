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

* Python 2.7 or Python 3.5
* Docker version 17 or later
    * [docker](https://docs.docker.com/install/)  
        - find and follow the instructions for your operating system

### Setup the Environment

1. In `Makefile` of the repository root directory, modify the environment variable definition on line
37.  It can be advantageous to mount a storage directory though a default option is available.     
2. Setup the development environment in a Docker container with the following command:
    - `make init`
    
    This command gets the resources for training and testing, and then prepares the Docker image for the experiments.
3. After creating the Docker image, run the following command.

- `make create-container`

    The above command creates a Docker container from the Docker image which we create with `make init`, and then
login to the Docker container.  This command needs to be run only once after creating the docker image.  After the
container hs been created with the command above, use the following command to enter the existing container: `make start-container`.

## Train the product model agent from scratch 
1. start the container:
    * `make start-container` or `make create-container`
2. Set the `config/model.json` "overwrite_experiment" parameter to "true".
3. run the following command from the docker container: 
    * `python3 ./scripts/train.py`

Voila! The script will load the in necessary pieces to run a training of the product model from 
scratch.  A tdqm progress bar should provide the necessary information to understand agent performance 
over the course of training.  


## Evaluate the product model

### Evaluate the checkpoints of MY trained model
I have stored the set of checkpoints from the final version of the trained product model inside 
of the base docker image.  To validate those checkpoints without having retrained the model locally, 
one may copy my version of the model with its checkpoints into the appropriate location, prior to 
running the evaluation script.  


1. start the container:
    * `make start-container` or `make create-container`.
2. run the command: `make mount-prodmod`
3. update the "evaluation_id" parameter in `config/eval.json`
4. run the following command from the docker container workdir:
    * `python3 ./scripts/train.py`

### Evaluate the checkpoints of your locally trained model
1. start the container:
    * `make start-container` or `make create-container`.
2. update the "evaluation_id" parameter in `config/eval.json`
3. run the following command from the docker container workdir:
    * `python3 ./scripts/train.py`
    
The script will load all of the checkpoints created during the specified model's
training, and will record the agent's score over the specified number of episodes.
These results are stored in the model's data folder, within the evaluation directory.  

## Credits

This repository was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [cookiecutter-docker-science](https://docker-science.github.io/) project template.
