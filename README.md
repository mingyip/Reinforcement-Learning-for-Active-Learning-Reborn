# Reinforcement-Learning-for-Active-Learning-Reborn

## Objective
In this project, we would like to apply reinforcement learning to perform active learning. We propose and implement a reinforcement learning algorithm that learns to select the desired subset of a dataset for deep classification learning such that the classification network can achieve a better accuracy at the final state. The Program is implemented in Python with Tensorflow

## Requirements
The environment is run in python 3.6, tensorflow 1.8.0 and Keras.

## Getting Started
To train an agent, run:
```
$ python train.py
```

In the default setting, the program trains a state-value estimation agent on the MNIST dataset. You can change the setting by uncommenting the options you need in **python/config.py**

```
    # GENERAL INFO
    VERSION = 1.1
    NAME = "dqn-with-exploration"
    DESCRIPTION = ""
    
    # AGENT TYPE
    AGENT_TYPE = "valueAgent"
    # AGENT_TYPE = "BVSB"
    # AGENT_TYPE = "random"

    # Env TYPE
    # ENV_TYPE = "emnist"
    # ENV_TYPE = "cifar"
    ENV_TYPE = "mnist"
    ....
```

## Run it on the server
To run many experiments as a batch on the server, you may use two helper shell scripts **generate.sh** and **run.sh**. The **generate.sh** file generates *config.py* files for later on training purposes. You may change the setting in the file to adopt your needs. E.g., If we want to run some experiments to study the effect of the classification budget ranging from 100 to 5000, you can change the idx array to set the parameter you need.
```
idx=(100 200 300 500 600 800 1000 1500 2000 2500 5000)

for t in ${idx[@]}; do
     echo "$t"
 
     sed -i -e 's/EVALUATION_CLASSIFICATION_BUDGET.*=.*/EVALUATION_CLASSIFICATION_BUDGET='"$t"'/g'     config.py
     cp config.py configs/run/config"$t".py
```
The generated configs are stored in the **python/configs/run** folder. You now and run the experiments with **run.sh** script.

## Folder Structure
The main program is located in the **python** folder. There are also some jupyter-notebook codes I implemented in the earilier state. 


If you want to add new datasets or agent methods, you may add them in **env/** and **agent/** folders respectively. Then you should also import your newly added files in **train.py**
```
Reinforcement-Learning-for-Active-Learning
    .
    ├── pictures/                   
    ├── presentation/               
    ├── python/                     
    │   ├── configs/
    │   ├── learningai/
    │   │   ├── agent               # The agent netowrks or agent implementation
    │   │   │   ├── bvsb.py
    │   │   │   ├── random.py
    │   │   │   └── value.py
    │   │   ├── env                 # The classification networks
    │   │   │   ├── cifar
    │   │   │   ├── emnist 
    │   │   │   └── mnist
    │   │   └── utils
    │   ├── result/                 # Results of runs
    │   ├── utils/                  # Some helper functions and the logger manager
    │   ├── config.py
    │   ├── train.py                # Train an agent
    │   ├── test.py                 # For unit test purpose
    │   ├── generate.sh
    │   └── run.sh                
    └── report/
```

## Dataset
We tested our algorithm in MNIST, CIFAR-10 and EMNIST dataset.

## Results
When a experiemnt is finished, all data is stored in .csv format. you may find the data in folder *result/finish/*

## References
<p> [1] Yann LeCun, Corinna Cortes, and CJ Burges. Mnist handwritten digit database.ATTLabs [Online]. 
Available: http://yann.lecun.com/exdb/mnist, 2, 2010.
</p>

<p> [2] Alex Krizhevsky. Learning multiple layers of features from tiny images. Technical report, 2009.
</p>

<p> [3] Gregory Cohen, Saeed Afshar, Jonathan Tapson, and Andre Van Schaik. Emnist: Ex-tending mnist to handwritten letters.2017 International Joint Conference on Neural Networks (IJCNN), 2017 </p>


If you find this implementation useful and want to cite/mention this page, here is a bibtex citation:

```bibtex
@misc{Idelbayev18a,
  author       = "Ming Yip Cheung",
  title        = "Reinforcement Learning for Active Learning",
  howpublished = "\url{https://github.com/mingyip/Reinforcement-Learning-for-Active-Learning-Reborn}",
  note         = "Accessed: 20xx-xx-xx"
}

```
