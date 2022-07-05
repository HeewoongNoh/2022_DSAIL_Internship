import yaml
from model_MF import BPR
from data import ml_100k, train, test
import matplotlib.pyplot as plt

#Reading configuration yaml file for hyperparamters
with open('configuration.yaml') as f:
    config = yaml.safe_load(f)

################################When using BPR based on model_MF##################################
train_data, test_data, gt_data = train, test, ml_100k
model_MF = BPR(gt_data,train_data,test_data,config['k'],config['learning_rate'],config['cost_parameter'])
model_MF.train(config['epochs'])


################################When using BPR based on model_kNN#################################



################################When using BPR based on model_MMMF################################
