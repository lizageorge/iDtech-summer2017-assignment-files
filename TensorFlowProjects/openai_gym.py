import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os
import gym

# Load an environment

env = gym.make("CartPole-v1")

# Print observation and action spaces

print(env.observation_space)
print(env.action_space)

# Make a random agent (comment this code out after making the neural network)

games_to_play = 10

for i in range(games_to_play):
    # Reset the environment
    obs = env.reset()
    episode_rewards = 0
    done = False

    while not done:
        # Render the environment so we can watch
        env.render()

        # Choose a random action
        action = env.action_space.sample()

        # Take a step in the environment with the chosen action
        obs, reward, done, info = env.step(action)
        episode_rewards += reward

    # Print episode total rewards when done
    print(episode_rewards)

# Close the environment
env.close()

# Create a discounting function

discount_rate = 0.95


def discount_normalize_rewards(rewards):
    discounted_rewards = np.zeros_like(rewards)
    total_rewards = 0

    for i in reversed(range(len(rewards))):
        total_rewards = total_rewards * discount_rate + rewards[i]
        discounted_rewards[i] = total_rewards

    discounted_rewards -= np.mean(discounted_rewards)
    discounted_rewards /= np.std(discounted_rewards)

    return discounted_rewards
# Build a policy gradient neural network

class Agent:
    def __init__(self, num_actions, state_size):
        initializer = tf.contrib.layers.xavier_initializer()

        self.input_layer = tf.placeholder(dtype=tf.float32, shape=[None, state_size])

        # Neural network starts here

        hidden_layer = tf.layers.dense(self.input_layer, 8, activation=tf.nn.relu, kernel_initializer=initializer)
        hidden_layer_2 = tf.layers.dense(hidden_layer, 8, activation=tf.nn.relu, kernel_initializer=initializer)

        # Output of neural net
        out = tf.layers.dense(hidden_layer_2, num_actions, activation=None)

        self.outputs = tf.nn.softmax(out)
        self.choice = tf.argmax(self.outputs, axis=1)

        # Training Procedure
        self.rewards = tf.placeholder(shape=[None, ], dtype=tf.float32)
        self.actions = tf.placeholder(shape=[None, ], dtype=tf.int32)

        one_hot_actions = tf.one_hot(self.actions, num_actions)

        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=one_hot_actions)

        self.loss = tf.reduce_mean(cross_entropy * self.rewards)

        self.gradients = tf.gradients(self.loss, tf.trainable_variables())

        # Create the operation to update gradients with the gradients placeholder.
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-2)
        self.update_gradients = optimizer.apply_gradients(zip(self.gradients_to_apply, tf.trainable_variables()))

# Create the training loop

tf.reset_default_graph()

# Modify these to match shape of actions and states in your environment
num_actions = 2
state_size = 4

path = "./cartpole-pg/"

training_episodes = 1000
max_steps_per_episode = 10000
episode_batch_size = 5

agent = Agent(num_actions, state_size)

init = tf.global_variables_initializer()

saver = tf.train.Saver(max_to_keep=2)

if not os.path.exists(path):
    os.makedirs(path)

with tf.Session() as sess:
    sess.run(init)

    total_episode_rewards = []

 # Create a buffer of 0'd gradients
    gradient_buffer = sess.run(tf.trainable_variables())
    for index, gradient in enumerate(gradient_buffer):
        gradient_buffer[index] = gradient * 0

    for episode in range(training_episodes):
        state = env.reset()

        episode_history = []
        episode_rewards = 0
# Close the environment
env.close()



