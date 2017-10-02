from gym_torcs import TorcsEnv

env = TorcsEnv(vision=False, throttle=True, gear_change=False)

print('testing environment')

for i in range(10):
    print("episode: ", i)

    if i % 3 == 0:
        env.reset(relaunch=True)
    else:
        env.reset()

    for i in range(100):
        # action = np.random.random(3)
        # print(action)
        action = [0.2, 1, 0]
        observe, reward, done, info = env.step(action)

        # print(observe.rpm)
        # print(observe.wheelSpinVel)
        # print(observe.track, done)
        # -1 ~ 1
        # print('the distance between car and track: ', observe.trackPos)
        # -1 : -180 , +1 : +180
        # print('the angle between car and track: ', observe.angle)
        # print("x speed of car: ", observe.speedX, " y speed of car: ",
        #       observe.speedY, " z speed of car: ", observe.speedZ)