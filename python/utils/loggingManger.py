
import warnings
import datetime
import csv
import os

from config import Config
from shutil import copy

class loggingManger(object):
    # DONE:  copy the config to the file
    # TODO: let agents to create its own log (eg. training.txt, selection.txt, agent_prediciton.txt, env_output.txt)
    # DONE:  check if folder "result" exists, if not create a folder

    def __init__(self, config_filename="config.py"):

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
        config_path = os.path.join(self.outpath_path, config_filename)
        print(self.folderName)

        if not os.path.isdir(self.masterFolder):
            os.mkdir(self.masterFolder)
        if not os.path.isdir(self.outpath_path):
            os.mkdir(self.outpath_path)
        if not os.path.isfile(config_path):
            copy(config_filename, self.outpath_path)

        with open(self.outpath_path+'/log.csv', mode='w') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            writer.writerow(['Datetime', str(now)])
            writer.writerow(['Project', projName])
            writer.writerow(['Total Episodes', agentEpisodes])
            writer.writerow(['Budget', budget, '', 'Evaluation Budget', Config.EVALUATION_CLASSIFICATION_BUDGET])
            writer.writerow(['Selection Batchsize', selection_batchsize, '', 'Evaluation Selection Batchsize', Config.EVALUATION_SELECTION_BATCHSIZE])
            writer.writerow(['Train Batchsize', training_batchsize, '', 'Evaluation Train Batchsize', Config.EVALUATION_TRAINING_BATCHSIZE])
            writer.writerow(['Epoch', epochs, '', 'Evaluation Epoch', Config.EVALUATION_CLASSIFICATION_EPOCH])
            writer.writerow(['Exploration Decay Rate', Config.EXPLORATION_DECAY_RATE])

    def __new__(cls):
        if not hasattr(cls, 'instance') or not cls.instance:
            cls.instance = super().__new__(cls)
        else:
            warnings.warn("loggingManger instance has already created.")

        return cls.instance

    def log(self, msg, filename=None, newline=False):
        """ write messages to log file """
        
        with open(self.outpath_path+'/log.csv', mode='a') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            if newline:
                writer.writerow('')
            writer.writerow(msg)
