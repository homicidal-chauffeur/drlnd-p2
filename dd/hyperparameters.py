PARAMS = {
    "ddpg": {
        "base": {
            "env_file": "/home/bohm/workspace/machine_learning/reinforcement_learning/udemy/Reacher_Linux/Reacher.x86_64",
            "lr": 1e-4,
            "batch_size": 64,
            "replay_size": 1000000,
            "replay_init": 10000,
            "gamma": 0.99,
            # for py-bullet it needs to be true
            "clip_actions": True,
            "stopping_reward": 32,

            "result": """
            """
        }
    },
    "d4pg": {
        "base": {
            "env_file": "/home/bohm/workspace/machine_learning/reinforcement_learning/udemy/Reacher_Linux/Reacher.x86_64",
            "lr": 1e-4,
            "batch_size": 64,
            # the paper used 1M
            "replay_size": 1000000,
            "replay_init": 100000,
            "gamma": 0.99,
            "reward_steps": 5,
            # for py-bullet it needs to be true
            "clip_actions": True,
            "stopping_reward": 32,

            "result": """
            """
        }
    },

}