
import warnings
import datetime
import os

from config import Config
from shutil import copy

class loggingManger(object):
    # DONE: copy the config to the file
    # TODO: let agents to create its own log (eg. training.txt, selection.txt, agent_prediciton.txt, env_output.txt)
    # DONE: check if folder "result" exists, if not create a folder

    def __init__(self):

        projName                = Config.NAME
        agentEpisodes           = Config.AGENT_TRAINING_EPISODES
        budget                  = Config.CLASSIFICATION_BUDGET
        selection_batchsize     = Config.SELECTION_BATCHSIZE
        training_batchsize      = Config.TRAINING_BATCHSIZE
        epochs                  = Config.CLASSIFICATION_EPOCH
        self.masterFolder       = Config.LOG_TOP_FOLDER
        
        now = datetime.datetime.now()
        timestamp = "{0:02d}{1:02d}{2:02d}-{3:02d}{4:02d}{5:02d}".format(
            now.year, now.month, now.day, now.hour, now.minute, now.second
            )
        config_info = "episodes({0})-budget({1})-selectionsize({2})-trainingsize({3})-epoch({4})".format(
            agentEpisodes, budget, selection_batchsize, training_batchsize, epochs
            )
        self.folderName = "{0}_{1}_{2}".format(timestamp, projName, config_info)
        self.outpath_path = os.path.join(self.masterFolder, self.folderName)
        config_path = os.path.join(self.outpath_path, "config.py")
        print(self.folderName)


        if not os.path.isdir(self.masterFolder):
            os.mkdir(self.masterFolder)
        if not os.path.isdir(self.outpath_path):
            os.mkdir(self.outpath_path)
        if not os.path.isfile(config_path):
            copy("config.py", self.outpath_path)

    def __new__(cls):
        if not hasattr(cls, 'instance') or not cls.instance:
            cls.instance = super().__new__(cls)
        else:
            warnings.warn("loggingManger instance has already created.")
                
        return cls.instance

    def log(self, filename, dict):
        pass