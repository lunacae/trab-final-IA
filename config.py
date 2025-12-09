treinando = False # Mude para False para testar o modelo/agent

#Troque os valores para treino ou exibição do modelo selecionado
class Config:
    # Reward configs
    STEP_REWARD = 0
    WALL_REWARD = 0
    HOLE_REWARD = 0
    GOAL_REWARD = 0

    # Q-Learning configs
    ALPHA = 0
    GAMMA = 0
    EPSILON = 0

    SIMMULATION_NUMBER = 50000 # Padrão para todos os modelos
    ALPHA_DECAY = 0
    EPSILON_DECAY = 0
    DECAY_STEP = 0

    # Train or Test Model/Agent
    # Set TRAIN to True and RENDERS to False to train the model/agent
    # Set TRAIN to False and RENDERS to True to test the model/agent
    TRAIN = treinando
    RENDERS = not treinando

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

def RENDERS() -> bool:
    return Config.RENDERS