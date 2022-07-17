# BPR  




# Methods  
 I used 3 methods(MF, kNN, WRMF) using BPR learner, metric: AUC  
 
## model_MF (matrix factorization)  
<img width="437" alt="bpr_mf_processing" src="https://user-images.githubusercontent.com/62690984/179381580-e3acd4c9-2610-4d62-83ed-092e23878f54.png">  
Same epoch with model_kNN (50000, it can be seen as much times due to bootstramp sampling), model_MF AUC score will be higher with more epoch.  
It is not in the overfitting.  

## model_kNN (k-Nearest Neighbor)
-including bootstrap sampling  
<img width="433" alt="model_kNN_without diag 0" src="https://user-images.githubusercontent.com/62690984/178977160-2670e33c-617f-4177-8903-0e60390c9f18.png">

## model_WRMF ( ALS + Confidence level matrix factorization)
<img width="440" alt="bpr_model_wrmf" src="https://user-images.githubusercontent.com/62690984/178976694-d3a6c966-59da-40b8-98f4-66fc0e95e2e3.png">
