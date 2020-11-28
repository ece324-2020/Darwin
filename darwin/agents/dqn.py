import collections
import numpy as np

# import tensorflow as tf

# from tf_agents.agents.dqn import dqn_agent
# from tf_agents.networks import q_network
# from tf_agents.trajectories import trajectory
# from tf_agents.replay_buffers import tf_uniform_replay_buffer
# from tf_agents.utils import common

# tf.compat.v1.enable_v2_behavior()


# Discount
GAMMA = 0.99
# Soft update
ALPHA = 0.5
# % chance of applying random action
EPSILON = 0.1

NUM_ITERATIONS = 2000
LEARNING_RATE = 1e-3

class DQNAgent:
    def __init__(self, env):
        r1, r2, r3 = env.action_space.spaces['action_movement'].spaces[0].nvec
        self.actions = [(x, y, z) for x in range(r1) for y in range(r2) for z in range(r3)]
        self.env = env

        # self.q_net = q_network.QNetwork(
        #     self.env.observation_spec(),
        #     self.env.action_spec(),
        #     fc_layer_params=(100,)
        # )

        # self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=LEARNING_RATE)

        # self.agent = dqn_agent.DqnAgent(
        #     self.env.time_step_spec(),
        #     self.env.action_spec(),
        #     q_network=self.q_net,
        #     optimizer=self.optimizer,
        #     td_errors_loss_fn=common.element_wise_squared_loss,
        #     train_step_counter=tf.Variable(0)
        # )

        # self.agent.initialize()

    


    def update(self, s, a, s_next, done):
        """
        Args:
        - s: current observation
        - a: current action
        - s_next: next observation
        """
        max_q_n = max([self.Q[s_n, a_n] for a_n in actions])
        # Don't include next state if done is 1 (if at final state)
        self.Q[s, a] += ALPHA * (r + GAMMA * max_q_n * (1 - done) - self.Q[s, a])

    def act(self, obs):
        # Force observation using epsilon-greedy
        if np.random.random() < EPSILON:
            return self.env.action_space.sample()
        
        action_val_pairs = [(a, self.Q[obs, a]) for a in actions]
        max_q = max(action_val_pairs, key=lambda x: x[1])
        action_choices = [a for a, q in action_val_pairs if q == max_q]
        # Select random action of action set with maximum reward value
        return np.random.choice(action_choices)
