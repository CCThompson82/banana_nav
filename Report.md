## Deep RL Nanodegree Project: Banana Collector Navigation

## Introduction

The objective of this project task was to implement a deep reinforcement learning solution to a 
simplified version of the Unity Engine ML-Agent Banana Collector game.  This report outlines and 
describes my solution, highlighting the important regions of code, results that demonstrate 
solution of the task,  as well as future directions that could be explored to make the solution of 
this task even faster.  

## Methods and Model Description
### Repository Architecture and Development Process
This solution was coded in an agile process, by which minimum viable product models were iterated 
until a full solution was achieved.  In order to aid in this process, a client model interface was 
generated, in which new versions of models could be loaded and run via a training script and its 
calls to a generic model API.  The process of training is observed in the file `./scripts/train.py`.

#### Training Script
The training script is generic and is responsible for interacting with a Model object API to perform
 the same set of steps for all model performance comparisons.  In brief, it is responsible for 
checking the repository configs settings to load a model version into the client model, and then 
interacts with that client's API in order to train, checkpoint, record performance, etc of the 
agent over the coarse of training.  

Upon running the script a progress bar is informative of the agent's current, and rolling average 
performance.  The scores of each episode are recorded, alongside the hyperparameters and 
configuration dictionaries that were used to launch the training.  

#### Evaluation Script
An evaluation script was utilized to compare the various checkpoints made during each model's 
training process, whereby network updates and random action choices are not utilised.  The result
of running the evaluation script is the recording of a large number of episode repetition scores for
each checkpoint of a model.  With this information, it is easy to compile a mean average episode 
score over the course of an agent's training.  

#### Development 
A series of model versions, designated `deepQN_v{}` were developed iteratively, with each new model
version using the previous as a sub-class.  Each version of the model was tested against a few 
important hyperparameters for the feature being added.

* deepQN_v0 - vanilla 3-layer DQN solution.  Experiences were batched in order of their occurrence.
* deepQN_v1 - added experience replay to the solution.  Experiences were accumulated into a buffer 
and batched randomly.  
* deepQN_v2 - reverts to no experience replay, but adds fixed-target network for estimation of 
action values
* deepQN_v3 - utilises both experience replay and fixed target estimation of action values

Once these models were used for hyperparameter exploration, as well as relative importance of each 
feature in performance, a final product level model was generated.  

##### Final Product Model

###### Base Network Architecture
The final product model solution consisted of a 3-layer neural network, which received a batch of 
state vectors (37-dimensional) as input, and yielded the action values for each state in the batch
upon forward propagation.  This network consisted of two hidden layers of 64 and 32 nodes, 
respectively.  This base architecture is set up using the class object defined in 
`src/base_networks/base_network.py`.  The size of the hidden layers is provided from the model 
parameters, which is specific to an experiment_id, e.g. `models/product/best_exp/params.json`.

###### Experience Replay
In some tasks, training a DQN model with experiences immediately as they are collected can create 
an unstable training process, whereby early actions get locked in, creating very little exploration
of the state space.  This particularly is present in situations where actions are highly correlated 
to state (e.g. moving left means an agent only explores the left side of the environment).  To 
combat this, a model can employ experience replay.  Experience replay dictates that instead of 
using every experience immediately as it is acquired for training, the model instead will build up 
a large buffer of experiences.  Once a sufficient buffer is created, the buffer is sampled randomly
to compose batches of experiences for training.  This process does not break the correlation 
between action and next state, but does prevent early training steps from failing to generate a 
diverse set of training experiences.

###### Fixed Target Action Value Estimation
DQN training is the process of backpropagating the error between the target action value and the 
expected action value of a state-action.  As both of these variables require feed-forward
propagation through the DQN, it is likely that errors will runaway if the same network is used for
both variables.  For instance, an early impossibly high estimation for a particular action value 
will continue to result in a large error when subsequent accurate action values are experienced. 
This large error creates a runaway gradient that destabilizes training. 

To combat this, a model may use fixed target action value estimation.  For this solutions, 
separate networks are used for the calculation of the current estimated action value ("target"), 
and the expected return for a given state-action.  After each training step, the fixed network is 
moved slightly in the direction of the estimation network by a factor of parameter tau.  

This process stabilizes training and avoids allowing early inaccurate action value experiences from 
causing a run-away gradient error.   

###### Final Parameters and Training
The final parameters used for the product/best_exp model are found in 
`models/product/best_exp/params.json`, and were determined empirically through experimentation 
during the development process.  

The final hyperparameters used to train the product model are available in 
`data/product/best_exp/experiment_id`.

## Results 

### Training performance

### Evaluation performance



## Discussion