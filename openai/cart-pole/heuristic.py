import gym

env = gym.make('CartPole-v0')

iters = 1000
episodes = 20
tR = 0

for _ in range(episodes):
    
    observation = env.reset()
    done = False

    t = 0
    while done != True:
        # env.render()
        
        action = env.action_space.sample()
        # action = int(observation[2] > 0 and observation[3] > 0)
        
        observation, reward, done, info = env.step(action)

        t += 1

        
    tR += t

print(tR / episodes)

env.close()