class Config:
    def __init__(self, args):
        # Reward configs
        self.STEP_REWARD = float(args.step_reward) if args.step_reward is not None else -0.1
        self.WALL_REWARD = float(args.wall_reward) if args.wall_reward is not None else -5
        self.HOLE_REWARD = float(args.hole_reward) if args.hole_reward is not None else -100
        self.GOAL_REWARD = float(args.goal_reward) if args.goal_reward is not None else -100

        # Q-Learning configs
        self.ALPHA = float(args.alpha) if args.alpha is not None else 0.0003
        self.GAMMA = float(args.gamma) if args.gamma is not None else 0.99
        self.EPSILON = float(args.epsilon) if args.epsilon is not None else 1
        self.ALPHA_DECAY = float(args.alpha_decay) if args.alpha_decay is not None else 0.999
        self.EPSILON_DECAY = float(args.epsilon_decay) if args.epsilon_decay is not None else 0.005
        self.DECAY_STEP = float(args.decay_step) if args.decay_step is not None else 1000

        # Train or Test Model/Agent
        self.TRAIN = str.lower(args.train) == "true"
        self.RENDER = str.lower(args.render) == "true"

    SIMMULATION_NUMBER = 50000
# def SIMMULATION_NUMBER() -> int:
#     return Config.SIMMULATION_NUMBER
