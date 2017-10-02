from gym_torcs import TorcsEnv
from collections import deque
import numpy as np
from keras.layers import Dense, Input, Add, Concatenate
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
import tensorflow as tf
import random


def ou_noise(x, mu, theta, sigma):
    return theta * (mu - x) + sigma * np.random.randn(1)


def normal(shape, scale=0.05, name=None):
    return K.variable(np.random.normal(loc=0.0, scale=scale, size=shape),
                      name=name)


class DDPGAgent:
    def __init__(self):
        self.action_size = 3
        self.state_size = 29

        self.actor, self.actor_weight = self.build_actor()
        self.actor_target, self.actor_target_weight = self.build_actor()
        self.critic, self.critic_state, self.critic_action = self.build_critic()
        self.critic_target, _, _ = self.build_critic()

        # actor optimizer
        self.action_grads = K.placeholder(shape=[None, self.action_size])
        params_grad = tf.gradients(self.actor.output, self.actor_weight,
                                   -self.action_grads)
        grads = zip(params_grad, self.actor_weight)
        self.optimize = tf.train.AdamOptimizer(0.0001).apply_gradients(grads)

        self.memory = deque(maxlen=100000)
        self.batch_size = 32
        self.discount_factor = 0.99
        self.tau = 0.001
        self.epsilon = 1
        self.epsilon_decay = 1/100000

        self.sess = tf.Session()
        K.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

    def build_actor(self):
        print("building actor network")
        input = Input(shape=[self.state_size])
        h1 = Dense(300, activation='relu')(input)
        h2 = Dense(600, activation='relu')(h1)
        steer = Dense(1, activation='tanh')(h2)
        accel = Dense(1, activation='sigmoid')(h2)
        brake = Dense(1, activation='sigmoid')(h2)
        action = Concatenate()([steer, accel, brake])
        actor = Model(inputs=input, outputs=action)
        return actor, actor.trainable_weights

    '''
    def actor_optimizer(self):
        self.action_grads = K.placeholder(shape=[None, self.action_size])
        # loss = -self.actor.output * action_grads
        params_grad = tf.gradients(self.actor.output, self.actor_weight,
                                   -self.action_grads)
        grads = zip(params_grad, self.actor_weight)
        self.optimize = tf.train.AdamOptimizer(0.0001).apply_gradients(grads)
        # optimizer = Adam(lr=0.0001)
        # updates = optimizer.get_updates(self.actor_weight, [], loss)
        # train = K.function([self.actor.input, action_grads], [], updates=updates)
        # return train
    '''

    def update_actor(self, states, gradient):
        self.sess.run(self.optimize, feed_dict={
            self.actor.input: states,
            self.action_grads: gradient
        })

    def build_critic(self):
        print("building critic network")
        state = Input(shape=[29])
        action = Input(shape=[3], name='action_input')
        w1 = Dense(300, activation='relu')(state)
        h1 = Dense(600, activation='linear')(w1)
        a1 = Dense(600, activation='linear')(action)
        h2 = Add()([h1, a1])
        h3 = Dense(600, activation='relu')(h2)
        V = Dense(1, activation='linear')(h3)
        model = Model(inputs=[state, action], outputs=V)
        model.compile(loss='mse', optimizer=Adam(lr=0.001))
        # model.summary()
        return model, state, action

    def get_action(self, state):
        self.epsilon -= self.epsilon_decay
        noise = np.zeros([self.action_size])
        action = self.actor.predict(state)[0]
        noise[0] = max(self.epsilon, 0) * ou_noise(action[0], 0.0, 0.60, 0.30)
        noise[1] = max(self.epsilon, 0) * ou_noise(action[1], 0.5, 1.00, 0.10)
        noise[2] = max(self.epsilon, 0) * ou_noise(action[2], -0.1, 1.00, 0.05)
        real = action + noise
        return real

    def save_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_model(self):
        mini_batch = random.sample(self.memory, self.batch_size)

        states = np.asarray([e[0] for e in mini_batch])
        actions = np.asarray([e[1] for e in mini_batch])
        rewards = np.asarray([e[2] for e in mini_batch])
        next_states = np.asarray([e[3] for e in mini_batch])
        dones = np.asarray([e[4] for e in mini_batch])

        target_q_values = self.critic_target.predict(
            [next_states, self.actor_target.predict(next_states)])

        targets = np.zeros([self.batch_size, 1])
        for i in range(self.batch_size):
            if dones[i]:
                targets[i] = rewards[i]
            else:
                targets[i] = rewards[i] + self.discount_factor * target_q_values[i]

        loss = 0
        loss += self.critic.train_on_batch([states, actions], targets)

        a_for_grad = self.actor.predict(states)

        action_grads = tf.gradients(self.critic.output, self.critic_action)
        grads = self.sess.run(action_grads, feed_dict={
            self.critic_state: states, self.critic_action: a_for_grad})[0]
        self.update_actor(states, grads)

        actor_weights = self.actor.get_weights()
        actor_target_weights = self.actor_target.get_weights()
        for i in range(len(actor_weights)):
            actor_target_weights[i] = self.tau * actor_weights[i] + \
                                       (1 - self.tau) * \
                                       actor_target_weights[i]
        self.actor_target.set_weights(actor_target_weights)

        critic_weights = self.critic.get_weights()
        critic_target_weights = self.critic_target.get_weights()
        for i in range(len(critic_weights)):
            critic_target_weights[i] = self.tau * critic_weights[i] + \
                                       (1 - self.tau) * \
                                       critic_target_weights[i]
        self.critic_target.set_weights(critic_target_weights)


agent = DDPGAgent()
env = TorcsEnv(vision=False, throttle=True, gear_change=False)

print('testing sample agent on torcs')
global_step = 0

for e in range(2000):
    step = 0
    score = 0
    if e % 10 == 0:
        observe = env.reset(relaunch=True)
        print("Now we save model")
        agent.actor.save_weights("ddpg_actor.h5", overwrite=True)
        agent.critic.save_weights("ddpg_critic.h5", overwrite=True)
    else:
        observe = env.reset()

    state = np.hstack((observe.angle, observe.track, observe.trackPos,
                       observe.speedX, observe.speedY, observe.speedZ,
                       observe.wheelSpinVel / 100.0, observe.rpm))
    done = False

    while not done:
        step += 1
        global_step += 1
        action = agent.get_action(state.reshape(1, state.shape[0]))
        observe, reward, done, info = env.step(action)
        score += reward
        next_state = np.hstack((observe.angle, observe.track, observe.trackPos,
                                observe.speedX, observe.speedY, observe.speedZ,
                                observe.wheelSpinVel / 100.0, observe.rpm))

        agent.save_sample(state, action, reward, next_state, done)

        if global_step > 1000:
            agent.train_model()

        # print(' step: ', step, ' action: ', action, ' reward: ', reward)
        state = next_state

        if done:
            print('episode: ', e, ' score: ', score, ' step: ', global_step,
                  ' epsilon: ', agent.epsilon)
