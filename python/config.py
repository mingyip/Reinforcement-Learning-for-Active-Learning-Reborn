class Config:

    # GENERAL INFO
    VERSION = 1.1
    NAME = "dqn-with-exploration"
    DESCRIPTION = ""
    
    # AGENT TYPE
    AGENT_TYPE = "valueAgent"
    # AGENT_TYPE = "BVSB"
    # AGENT_TYPE = "random"

    # Env TYPE
    ENV_TYPE = "emnist"
    # ENV_TYPE = "cifar"
    # ENV_TYPE = "mnist"
    
    # TRANING SETTING
    AGENT_TRAINING_EPISODES = 10
    CLASSIFICATION_BUDGET   = 1000
    CLASSIFICATION_EPOCH    = 10
    SELECTION_BATCHSIZE     = 100
    TRAINING_BATCHSIZE      = 10
    TRAINING_START_AT_ITER  = 0
    TRAINING_FINISH_AT_ITER = 4
    USE_EXPERIENCE_REPLAY   = True

    # HYPER-PARAMETER TUNNING 
    EXPLORATION_DECAY_RATE = 0.0025

    # VALIDATION SETTING
    EVALUATION_EPISODES                = 10
    EVALUATION_CLASSIFICATION_BUDGET   = 1000
    EVALUATION_CLASSIFICATION_EPOCH    = 10
    EVALUATION_SELECTION_BATCHSIZE     = 100
    EVALUATION_TRAINING_BATCHSIZE      = 10
    EVALUATION_START_RANK              = 0
    # Stream vs Pool base
    EVALUATION_IS_STREAM=False


    # SEED
    TF_SEED = 30
    NP_SEED = 30

    # loggerManger
    LOG_TOP_FOLDER = "result"

    # MEMORY_SIZE = 100000
    # NUM_LAST_FRAMES = 4
    # LEVEL = "snakeai/levels/10x10-blank.json"
    # NUM_EPISODES = -1
    # BATCH_SIZE = 64
    # DISCOUNT_FACTOR = 0.95
    # USE_PRETRAINED_MODEL = False
    # PRETRAINED_MODEL = "dqn-00000000.model"
    # # Either sarsa, dqn, ddqn
    # LEARNING_METHOD = "dqn"
    # MULTI_STEP_REWARD = False
    # MULTI_STEP_SIZE = 5
    # PRIORITIZED_REPLAY = False
    # PRIORITIZED_RATING = 1
    # DUEL_NETWORK = False
    # #foodspeed =0 no movement. foodspeed =2 food moves one step every 2 timesteps
    # FOODSPEED = 0