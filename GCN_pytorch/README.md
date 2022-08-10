## Semi-Supervised Classification with Graph Convolutional Networks(Thomas N.Kipf & Max Welling, 2016)  
Scalable approach for semi-suprvised learning on graph-structured data which is based on an efficient variant of CNN that operate directly on graphs  
-First-order approximation of localized spectral filters on graph  
-GCN can encode both graph sturcture and node features in a way useful for semi-supervised classification  

## Model  
Proposed renormalized model(using renormalization trick) offers both improbed efficiency(fewer parameters and operations)
and better predictive performance on a number of dataset to other models  
### Summary  
![gcn_ppt_slides_capture](https://user-images.githubusercontent.com/62690984/183788800-9ab159d1-6374-446c-a3a5-baa7bc1b287d.png)  
### Model Sturcture  
<img width="533" alt="gcn_archi_image" src="https://user-images.githubusercontent.com/62690984/183788172-f0023139-8030-4b66-9e6b-e1d3fa285a5f.png">  



## Experiments & Results
I ran 2-layer GCN, 3-layer GCN and reported each model's train accuracy and validation accuracy with Tensorboard  
<img width="574" alt="implementation" src="https://user-images.githubusercontent.com/62690984/183785110-cd370b3f-3151-4da7-b570-28c20465e912.png">      
2-lyaer GCN model resulted better performance on the Cora dataset (Node classification accuracy, Epoch:200)    
<img width="545" alt="tensorboard_legend" src="https://user-images.githubusercontent.com/62690984/183785142-36507321-c581-4cbd-b823-6ee4a5f05f7c.png">  

## References
[1][Kipf & Welling,Semi-Supervised Classification with Graph Convolutional Networks, 2016](https://arxiv.org/abs/1609.02907)  
[2][Official code from Thomas Kipf](https://github.com/tkipf/pygcn)
