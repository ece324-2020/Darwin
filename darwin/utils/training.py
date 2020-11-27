import gym
# Base training loop
def enter_train_loop(env, policy, episodes):
    assert isinstance(env.action_space, gym.spaces.Discrete)
    assert isinstance(env.observation_space, gym.spaces.Discrete)
    for episode in range(episodes):

        observation = env.reset()
        done = False
        while not done:
            action = policy.act(observation)
            next_observation, r, done, info = env.step(action)
            env.render()
            policy.update(observation, action, next_observation, done)
            
            reward += r
            if done:
                rewards.append(reward)
                print('Finished episode, reward:', reward)
                reward = 0.
                observation = env.reset()
            else:
                observation = next_observation

        
    env.close()

