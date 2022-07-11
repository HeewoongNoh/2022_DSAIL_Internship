import yaml
import time
from dataset import ml_100k, train, test
from model_MF import BPR_MF
from model_kNN import BPR_kNN
from model_WRMF import WRMF
import matplotlib.pyplot as plt

#Reading configuration yaml file for hyperparamters
with open('configuration.yaml') as f:
    config = yaml.safe_load(f)

# YOU CAN CHOOSE BPR_MR, BRP_kNN, WRMF WHICH RANKED IN TOP3, METRIC: AUC
# Load data
train_data, test_data, gt_data = train, test, ml_100k

################################When using BPR based on model_MF##################################
start = time.time()
model_MF = BPR_MF(gt_data,train_data,test_data,config['k'],config['learning_rate'],config['cost_parameter'])
model_MF.train(config['epochs'])
end = time.time() - start
print(f'total{end} time elapsed')

################################When using BPR based on model_kNN#################################
start = time.time()
model_kNN = BPR_kNN(gt_data,train_data,test_data,config['learning_rate'],config['reg_pos'],config['reg_neg'])
model_kNN.train(config['epochs'])
end = time.time() - start
print(f'total{end} time elapsed')

################################When using BPR based on model_WRMF###############################
start = time.time()
model_WRMF = WRMF(gt_data,train_data,test_data,config['k'],config['alpha'],config['cost_parameter'])
model_WRMF.train(config['epochs'])
end = time.time() - start
print(f'total{end} time elapsed')
