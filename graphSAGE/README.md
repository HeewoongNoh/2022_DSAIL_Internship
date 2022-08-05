# GraphSAGE
Inductive Representation Learning on Large Graphs(NIPS 2017, Will Hamilton, Zhitao Ying, Jure Leskovec)  
Extension GCN to Inductive unsupervised learning, using trainable AGGREGATION functions  
**SAGE**:**SA**mple the required neighborhoods sets and aggre**G**at**E**


## Model 
![GraphSAGE_algorithm_pseudocode](https://user-images.githubusercontent.com/62690984/182365496-cb635672-5cd0-490f-a668-b4de5f3856a3.png)  
Aggregate functions can be any fuction that is symmetric, permutation invariant.  
Mean, Pooling, LSTM aggregator etc.  
For using LSTM which is not a permutation invariant function, applying random permutation of the node's neighbors is necessary  
LSTM can give an advantage of larger expressive capability  
## Experiment & Result -Node Classification-
### CORA Dataset (standard citation network benchmark dataset)
I tried to implement general GNN including graphSAGE layer which uses mean aggregator.  
For the Cora dataset, there are 2708 nodes, 5429 edges, 7 prediction classes for nodes, and 1433 features per node.  
Epoch: 500, learning_rate: 0.01  
![github_training_process](https://user-images.githubusercontent.com/62690984/182365698-7fa49ebf-72bf-45f9-a34f-f066a8b1518e.png)  
### Training loss & Test accuracy  
![github_epoch500_lr_0 01_cora_SAGE](https://user-images.githubusercontent.com/62690984/182366365-e840b89c-e190-4ad3-9050-e3c3ce326ab8.png)  
## References 
[1] "Inductive Representation Learning on Large Graphs",https://arxiv.org/abs/1706.02216
