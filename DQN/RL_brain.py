"""
This part of code is the DQN brain, which is a brain of the agent.
All decisions are made in here.
Using Tensorflow to build the neural network.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Using:
Tensorflow: 1.0
gym: 0.7.3
"""

import numpy as np
import random
from dataclasses import dataclass
import tensorflow as tf
from tensorflow import keras
from typing import List

np.random.seed(1)
tf.random.set_seed(1)

@dataclass
class Step:
    __slots__ = ['current_state', 'action', 'reward', 'next_state']
    current_state: List[int]
    action: int
    reward: float
    next_state: List[int]

    # def to_array(self):
    #     return np.hstack([self.current_state, [self.action, self.reward], self.next_state])


class Memory:
    """ Used to store historical steps """

    def __init__(self, memory_size: int):
        self.memory_size = memory_size
        self.memory_count = 0
        self.memory: List[Step] = []

    def record_step(self, current_state: List[int], action: int, reward: float, next_state: List[int]):
        s = Step(current_state, action, reward, next_state)

        if self.memory_count < self.memory_size:
            self.memory.append(s)
        else:
            self.memory[self.memory_count % self.memory_size] = s
        self.memory_count += 1

    def sample_steps(self, batch_size: int) -> dict:
        if self.memory_count > batch_size:
            steps = np.random.choice(self.memory, size=batch_size, replace=False)
        else:
            steps = self.memory

        return {
            'current_state': np.vstack([s.current_state for s in steps]),
            'action': np.vstack([s.action for s in steps]),
            'reward': np.vstack([s.reward for s in steps]),
            'next_state': np.vstack([s.next_state for s in steps]),
        }


class Net(keras.models.Model):

    def __init__(self, n_action: int, **kwargs):
        super().__init__(**kwargs)
        self.n_action = n_action    # size of action space
        self.d1 = keras.layers.Dense(20, activation='relu')
        self.d2 = keras.layers.Dense(self.n_action, activation=None)

    def call(self, inputs, training=None, mask=None):
        out = self.d1(inputs)
        out = self.d2(out)
        return out


class DQN:

    def __init__(self,
                 n_action: int,
                 memory_size: int = 500,
                 batch_size: int = 32,
                 reward_decay: float = 0.9,
                 e_greedy: float = 0.9,
                 replace_target_iter: int = 300):
        self.n_action = n_action
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.reward_decay = reward_decay
        self.e_greedy = e_greedy
        self.replace_target_iter = replace_target_iter

        self.eval_net = Net(n_action=n_action)
        self.target_net = Net(n_action=n_action)
        self.memory = Memory(memory_size=memory_size)

        self.optimizer = tf.optimizers.RMSprop(learning_rate=1e-2)
        self.loss_fn = tf.losses.MeanSquaredError()
        self.learn_step_count = 0

    def sync_weight(self):
        """ Update the weight in target_net using the weights from eval_net"""
        self.target_net.set_weights(self.eval_net.get_weights())

    def choose_action(self, observation):
        if random.random() < self.e_greedy:
            q_value = self.eval_net(observation[np.newaxis, :])
            return np.argmax(q_value[0])
        else:
            return random.randint(0, self.n_action-1)

    def memorize(self, current_state, action, reward, next_state):
        self.memory.record_step(current_state, action, reward, next_state)

    def learn(self):
        batch_sample = self.memory.sample_steps(self.batch_size)

        with tf.GradientTape() as tape:

            q_next = self.target_net(batch_sample['next_state'])
            q_current = self.eval_net(batch_sample['current_state'])

            # the calculation process is same as in Q-Learning
            q_target = q_current.numpy()
            batch_index = np.arange(q_target.shape[0], dtype=np.int32)
            action_index = batch_sample['action'].astype(int)
            q_target[batch_index, action_index] = batch_sample['reward'] + np.max(q_next, axis=1) * self.reward_decay

            loss = self.loss_fn(q_target, q_current)
            print(f'Step: {self.learn_step_count}: {loss}')

        t_vars = self.eval_net.trainable_variables
        gradients = tape.gradient(loss, t_vars)
        self.optimizer.apply_gradients(zip(gradients, t_vars))

        if self.learn_step_count % self.replace_target_iter == 0:
            self.sync_weight()
        self.learn_step_count += 1
