class Config:
    # Reward configs
    STEP_REWARD = -0.1
    WALL_REWARD = -5
    HOLE_REWARD = -100
    GOAL_REWARD = 100

    # Q-Learning configs
    ALPHA = 0.0003
    GAMMA = 0.99
    EPSILON = 1
    SIMMULATION_NUMBER = 10
    ALPHA_DECAY = 0.999
    EPSILON_DECAY = 0.005
    DECAY_STEP = 1000

    # Train or Test Model/Agent
    TRAIN = True
    RENDERS = True

def STEP_REWARD() -> int:
    return Config.STEP_REWARD

def WALL_REWARD() -> int:
    return Config.WALL_REWARD

def HOLE_REWARD() -> int:
    return Config.HOLE_REWARD

def GOAL_REWARD() -> int:
    return Config.GOAL_REWARD

def ALPHA() -> float:
    return Config.ALPHA

def GAMMA() -> float:
    return Config.GAMMA

def EPSILON() -> float:
    return Config.EPSILON

def SIMMULATION_NUMBER() -> int:
    return Config.SIMMULATION_NUMBER

def ALPHA_DECAY() -> float:
    return Config.ALPHA_DECAY

def EPSILON_DECAY() -> float:
    return Config.EPSILON_DECAY

def DECAY_STEP() -> int:
    return Config.DECAY_STEP

def TRAIN() -> bool:
    return Config.TRAIN

def set_train(new_value):
    Config.TRAIN = new_value

def RENDERS() -> bool:
    return Config.RENDERS