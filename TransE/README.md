# Download
Download dataset for FB15K - 237  (can't upload large files)  
https://deepai.org/dataset/fb15k-237

# What is difference between FB15k and FB15k-237  
FB15k-237 is a link prediction dataset created from FB15k. While FB15k consists of 1,345 relations, 14,951 entities, and 592,213 triples, many triples are inverses that cause leakage from the training to testing and validation splits. __FB15k-237 was created by Toutanova and Chen (2015) to ensure that the testing and evaluation datasets do not have inverse relation test leakage.__ In summary, FB15k-237 dataset contains 310,079 triples with 14,505 entities and 237 relation types.   

# Problems that I met...
## nan loss
<img width="554" alt="lr_0 01_loss_nan" src="https://user-images.githubusercontent.com/62690984/178978472-02e18160-c802-4c80-9a8a-e9894f6565a2.png">
I tried 2 ways to solve nan loss  

1) with torch.autograd.detect_anomaly():  

2) use loss when it is not the NAN loss  

By 2), i could solve nan loss problem but i have trouble with training model.  
## Training?     
![epoch100_losses_transe](https://user-images.githubusercontent.com/62690984/178979987-1d37456b-9190-4e5e-b785-dd07596628a1.png)   
(Wihtout CUDA, there is a limitation on the number of epochs)  
I tried training the model on the same conditions metioned in the paper except for dataset ( FB15k --> FB15K-237)  
Trouble in training...i will keep fixing it....

