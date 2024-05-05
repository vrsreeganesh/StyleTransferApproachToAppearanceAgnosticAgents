# A Style Transfer Approach To Appearance Agnostic Agents

## Abstract
Sim2Real is an approach in robotics where an agent is trained in simulation and deployed in reality. In this project, we investigate Neural Style Transfer as a domain randomization approach to Sim2Real. We present the effectiveness of Neural Style Transfer and how it fares against other SOTA approaches used for Sim2Real.

## Contents
This repository contains the code for *A style Transfer Approach to Appearance Agnostic Agents*. To reproduce results, follow the following set of steps

## Setup
Unzip the two zip files for dataset creation scripts and training (and evaluation)  scripts
Clone the repo, setup CARLA 0.9.10.1, and build the conda environment as follows:
```
git clone 
conda env create -f environment.yml
conda activate tfuse
```



  



Most of the word was done on SCC. So all the files are present there.
However, I have copied the main scripts into this repository. 

Training Code:  
  train02.py  
    This is the python script that trains the transfuser model.  
  RunTrain02.sh   
    This ist he shell script that takes care of overall training and submits the training as a batch job on SCC.   
