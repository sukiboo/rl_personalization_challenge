# Reinforcement Learning Personalization Challenge
In this challenge your goal is to train an RL agent to solve a synthetic personalization task represented by the contextual bandit environment.

## Setup
Install the dependancies with `pip install -r requirements.txt`, then run with `python main.py`.

## Files
Essentially this simple repository consists of the following files:
* `environment.py` --- contains the class `SampleEnv` that creates an OpenAI Gym contextual bandit environment
* `main.py` --- trains a policy gradient agent, serving as a basic baseline --- **modify this file to implement and train your agent**

## Environment
The generated `SampleEnv` environment inherits from `gym.Env` and, as such, has the following methods:
* `reset()` --- observe a new state
* `step(action)` --- take an action and return the result

The above methods are technically sufficient to solve the environment.
Other useful methods include
* `evaluate_agent(agent)` --- compute the *deterministic* performance of the agent's policy on the environment
* `restart()` --- fully recreate the environment; should be called between the training of different agents for reproducibility
* `observe(num=1)` --- observe new states; identical to `reset` but can sample multiple states (`num`) simultaneously
* `compute_reward(s,a_ind)` --- compute the *normalized* reward for a state `s` and an action index `a_ind`
* `compute_reward_raw(s,a)` --- compute the *un-normalized* reward value of a state-action pair `(s,a)`
* `print_action_histogram()` --- print the histogram of the optimal actions; ideally an agent should provide a similar histogram

## Results
By default the reward values returned by the environment are *normalized*, i.e. the optimal reward for any state `s` is `1` and the average reward is `0`.
Hence any sensible agent should achieve a positive return and the optimal agent has the return of `1`.
For example, the current baseline agent achieves a performance score of `0.2318`.

The intended outcome is to train an agent that demonstrates a *good* performance, e.g. `> 0.8` or so.
If you manage to obtain such an agent, please let me know!
