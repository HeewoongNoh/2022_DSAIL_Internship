# Collaborative Metric Learning  
CML learns a joint user-item metric to emcode user's preferences as well as user-user and item-item similarity  
Collaborative filtering for implicit feedback and CML captures such relationships well.  

## Dataset
-Movielens latest small version  
-Included ratings greater or equal to 4 as positive feedback and transform as implicit feedback  
-Preprocessed moives.csv to get item's feature  
## Model  
-Using mlp(multi-layer-perceptron) with 256 hidden layers for using item features  
-Approximate Ranking Weight (rank_d_ij)  
-Loss = margin_loss + feature_loss + covariance_loss  
## Results
-Training  
<img width="617" alt="train_process_github" src="https://user-images.githubusercontent.com/62690984/179345430-b375dbd7-64b5-472e-b2ba-276f4d766c7d.png">  
-Training loss, recall#50 for train dataset and validation dataset(epoch:200)  
![readme for github](https://user-images.githubusercontent.com/62690984/179345466-f8932ab2-930a-484a-b08c-8cef807ab080.png)
