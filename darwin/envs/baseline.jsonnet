{
    make_env: {
        "function": "envs.baseline:make_env",
        
        args: {
            # Agents
            n_agents: 2,
            # Agent Actions
           
            # Scenario
            door_size: 6,
            horizon: 50,
            scenario: 'uniform',
        

            # Food
            n_food: 3,

            
            # Observations
            n_lidar_per_agent: 10,
           
        },
    },
}
