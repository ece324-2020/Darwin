from agents.q import QAgent
from agents.dqn import DQNAgent


# Instantiate policies that we work with
policies = {
    'q': lambda x: QAgent(x),
    'dqn': lambda x: DQNAgent(x)
}
