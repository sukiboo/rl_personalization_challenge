
import numpy as np
import tensorflow as tf

from environment import SampleEnv

np.set_printoptions(precision=4, suppress=True)
tf.random.set_seed(2022)


def evaluate_agent(agent, env, num_s=100000):
    '''evaluate the agent's performance on the randomly sampled states'''
    S = env.observe(num=num_s)
    A = agent(S).numpy().argmax(axis=1)
    R = env.compute_reward(S,A)
    print(f'\nagent evaluation reward = {R.mean():.4f}')
    hist = np.histogram(A, bins=np.arange(env.num_a+1), density=True)[0]
    print(f'agent action selection histogram:\n{-np.sort(-hist)}')


if __name__ == '__main__':

    # create environment
    env = SampleEnv()

    # create an agent
    agent = tf.keras.Sequential([tf.keras.Input(shape=(None,env.dim_s)),
                                 tf.keras.layers.Dense(256, activation='relu'),
                                 tf.keras.layers.Dense(128, activation='relu'),
                                 tf.keras.layers.Dense(64, activation='relu'),
                                 tf.keras.layers.Dense(env.num_a, activation=None)])

    # select optimization algorithm
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    # interact with the environment
    rewards, losses = [], []
    for t in range(10000):

        with tf.GradientTape() as tape:
            # observe a new state from the environment
            state = tf.reshape(tf.convert_to_tensor(env.reset(), dtype=tf.float32), shape=[-1,env.dim_s])

            # select an action by sampling agent's output
            action_probs = tf.nn.softmax(agent(state))
            logits = tf.math.log(action_probs)
            action_index = tf.random.categorical(logits, num_samples=1).numpy().item()

            # get reward from the environment
            _, reward, _, _ = env.step(action_index)
            # compute the loss
            loss = -reward * action_probs[0,action_index]

            # record training data
            rewards.append(reward)
            losses.append(loss)

        # train the agent
        grads = tape.gradient(loss, agent.trainable_variables)
        optimizer.apply_gradients(zip(grads, agent.trainable_variables))

        # report average performance
        if (t+1) % 100 == 0:
            print(f'iteration {t+1:6d}:'
                  + f'   reward = {np.mean(rewards[-1000:]): .4f},'
                  + f'   loss = {np.mean(losses[-1000:]): .2e}')

    # evaluate trained agent
    evaluate_agent(agent, env)

