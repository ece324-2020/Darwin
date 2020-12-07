# Project Darwin

Survival-based multi-agent reinforcement learning exploration.

The environments built in this repository depend on the MuJoCo physics engine.
Once a license is obtained and requirements are installed, run the application
to train an model on an environment, or view an environment, e.g.:

python3 darwin/runner.py darwin/envs/simple_food.py --policy-name=dqn

Live demos are available on [darwin-app.web.app](https://darwin-app.web.app/showcase/baseline1).

Environment building is inpsired by the environment design done by OpenAI for their
paper [Emergent Tool Use From Multi-Agent Autocurricula](https://arxiv.org/abs/1909.07528).
