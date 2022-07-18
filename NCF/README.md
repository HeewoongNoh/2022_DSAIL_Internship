# Neural Collaborative Filtering  
## Summary  
![NCF_Model_img](https://user-images.githubusercontent.com/62690984/179514077-7d3997af-97f8-4698-b508-1829e54ac808.jpg)  

- Neural network architecture to model latent features of users and items  
- General framework for collaborative filtering based on neural netwroks  
- MF(matrix factorization) can be interpreted as a special case of NCF, utilizing a multi-layer perceptron(mlp) for modeling high level of non-linearities.  
- GMF(Genralized Matrix Factorization) + MLP(Multi Layer Perceptron) : NeuMF  

## Model  
Using Dataset 'movielens-1m', I made 3 models GMF, MLP, NeuMF.  
I  pretrained 'GMF', 'MLP' model for 20 epochs respectively (MLP has 3 layers, latent dim:32, neg_sampling:4, learning_rate: 0.001,batch_size:256)  
Wit CUDA, I could run each model with more epochs expecting higher performance.  
I set the alpha:0.5, hyperparamter which determines the trade-off between the two pretrained model(GMF, MLP).  
Finally, I trained NeuMF model using pretrained weights from epoch20 GMF, epoch20 MLP  
(on the same configurations with pretrained models except for epoch , 30 epoch for:NeuMF)  

## Results  
### Training process of NeuMF  
<img width="444" alt="NeuMF_1" src="https://user-images.githubusercontent.com/62690984/179512601-58cae4f7-e1c2-46bf-b483-ffb71fa98535.png">  
...  
<img width="454" alt="NeuMF_2" src="https://user-images.githubusercontent.com/62690984/179512645-8c43741a-62dc-4c5f-8f69-8d74d1d6445a.png">  

### Training loss & Results (Metric: HR@10, NDCG@10)   

![github_epoch_30_NeuMF](https://user-images.githubusercontent.com/62690984/179513148-e6a50aef-4014-42ca-9097-47e2bc59cd60.png)
