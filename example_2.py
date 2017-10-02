from gym_torcs import TorcsEnv
from collections import deque
import numpy as np
from keras.layers import Dense, Input, merge
from keras.initializations import normal
from keras.models import Model


class DDPGAgent:
    def __init__(self):
        self.actor = self.build_actor()
        self.memory = deque(maxlen=100000)

    def build_actor(self):
        print("build actor network")
        input = Input(shape=[29])
        h1 = Dense(300, activation='relu')(input)
        h2 = Dense(600, activation='relu')(h1)
        steer = Dense(1, activation='tanh', init=lambda shape,
                      name: normal(shape, scale=1e-4, name=name))(h2)
        accel = Dense(1, activation='sigmoid', init=lambda shape,
                      name: normal(shape, scale=1e-4, name=name))(h2)
        brake = Dense(1, activation='sigmoid', init=lambda shape,
                      name: normal(shape, scale=1e-4, name=name))(h2)
        action = merge([steer, accel, brake], mode='concat')
        actor = Model(input=input, output=action)
        return actor

    def get_action(self, state):
        action = self.actor.predict(state)[0]
        return action


agent = DDPGAgent()
env = TorcsEnv(vision=False, throttle=True, gear_change=False)

print('testing sample agent on torcs')

for i in range(10):
    if i % 3 == 0:
        observe = env.reset(relaunch=True)
    else:
        observe = env.reset()

    state = np.hstack((observe.angle, observe.track, observe.trackPos,
                       observe.speedX, observe.speedY, observe.speedZ,
                       observe.wheelSpinVel / 100.0, observe.rpm))
    state = np.reshape(state, [1, np.shape(state)[0]])
    done = False

    while not done:
        action = agent.get_action(state)
        observe, reward, done, info = env.step(action)
        next_state = np.hstack((observe.angle, observe.track, observe.trackPos,
                                observe.speedX, observe.speedY, observe.speedZ,
                                observe.wheelSpinVel / 100.0, observe.rpm))
        next_state = np.reshape(next_state, [1, np.shape(next_state)[0]])

        state = next_state