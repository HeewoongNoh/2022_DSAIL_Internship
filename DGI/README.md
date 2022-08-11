## Deep Graph Infomax  
### Model overview  
![dgi_images](https://user-images.githubusercontent.com/62690984/184087795-728b00af-e4da-47cd-a776-88c7b5ba5a46.png)  

Learning the enocder relies on maximizing local mutual information. To obtain the graph-level summary vectors, leverage a readout function  
DGI does not rely on random walk objectives, and is readily applicable to both transductive and inductive learning.
Presenting a new approach for learning unsupervised representations on graph-structured data with competitive performance across both transductive and inductive classification  
## Experiments & Results
### Transductive setting  
I ran the trasductive_dgi on 'CORA' dataset.  
![transductive_dgi_training_loss](https://user-images.githubusercontent.com/62690984/184089404-2392042e-d53d-46d8-995a-ad62f89c39f5.png)  
![transductive_dgi_acc](https://user-images.githubusercontent.com/62690984/184089452-d00883b9-1600-42db-bbb8-fb71c88877e0.png)    

### Inductive setting  
I tried to training inductive_dgi on 'Reddit' dataset. Reddit is too large to run experiment without 'cuda'.  
Therefore, i could't train the model for many epochs.Below plot show training loww of DGI on Reddit dataset for 15 epoch, it seems unstable  
I will try training model again by sampling dataset or using smaller dataset.  
![Figure_1](https://user-images.githubusercontent.com/62690984/184089345-f5facdff-64fc-4ab6-815a-14ba24422e0d.png)  


## Reference  
[1][Deep Graph Infomax, Petar Veličković, 2019 ICLR](https://arxiv.org/abs/1809.10341)  
[2][Pytorch Geometric reference](https://pytorch-geometric.readthedocs.io/en/latest/modules/root.html)  
