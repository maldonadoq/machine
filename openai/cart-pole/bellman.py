import matplotlib.pyplot as plt
import numpy as np
import gym

env = gym.make("Taxi-v3")

Q = np.zeros([env.observation_space.n, env.action_space.n])
alpha = 0.618
episodes = 500
rewards = []

for episode in range(episodes):
    done = False
    
    G, reward = 0, 0
    state = env.reset()
    
    # env.render()
    while done != True:
        action = np.argmax(Q[state])
        state2, reward, done, info = env.step(action)
        Q[state,action] += alpha * (reward + np.max(Q[state2]) - Q[state,action])
        G += reward

        state = state2
    
    rewards.append(G)

plt.plot(range(episodes), rewards)
plt.show()

env.close()