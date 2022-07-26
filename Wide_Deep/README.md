# Wide & Deep learning for Recoomender Systems, Heng-Tze Cheng et al, Google Inc(DLRS, 2016)

## Data  
__Task Purpose:__ Classification whether user redeems the voucher (label: 0 or 1)  
I used "Lazada (Alibaba Group) e-commerce voucher redeem dataset which sampled from large scaled production dataset.    
You can get dataframe from  "kdd_data.pkl.zip".(from references).  
The dataset contains three dataframes corresponding users' voucher collection logs, related user behavior logs and related item features stored in a pickle  (.pkl) file .  
I made these dataframes to one, and preprocessed for using Wide, Deep model. (with Data_Preprocessing.ipynb)  
You can also get from df file (train, test) from my github/repository/data_files.  
"df_deep_final.pkl" for deep model, "df_wide_encoding_promotionID" that one-hot-encoded promotion ID feature, "df_wide_final.pkl" is not one-hot-encoded for promotion ID feature.  


## Model
### Wide component  
<img width="173" alt="wide_part" src="https://user-images.githubusercontent.com/62690984/181127390-18bef2f2-56a4-42e7-bc3a-43efea7cf668.png">  

### Deep Compoent
<img width="241" alt="Deep_part" src="https://user-images.githubusercontent.com/62690984/181127443-85f5d5bb-8bbb-4c74-83ee-d4d83d028260.png">  

### Joint Training & Overall structure for Wide & Deep Model  
<img width="542" alt="Joint Training" src="https://user-images.githubusercontent.com/62690984/181127593-9884b25f-db18-4643-aaf5-08118059f987.png">  

### Usage  
You can use pretrained_model from checkpoints files.   
(0) Data_Preprocessing (if you want to modify dataframe)  
(1) Dataset.py using "df_wide,deep_pro.pkl"   
(2) Model.py    
(3) Train.py(training for wide_only, deep_only and wide & deep model)  
(4) Main_WD.py for testing
## Result  
<img width="898" alt="결과 리드미" src="https://user-images.githubusercontent.com/62690984/181127677-5361b7ec-2cee-42e4-94ff-4f54b43477ae.png">

## References  
[Paper] (Heng-Tze Cheng, Levent Koc, Jeremiah Harmsen, Tal Shaked, Tushar Chandra,Hrishi Aradhye, Glen Anderson, 
Greg Corrado, Wei Chai, Mustafa Ispir, RohanAnil, Zakaria Haque, Lichan Hong, Vihan Jain, Xiaobing Liu, and 
Hemal Shah.2016. Wide & Deep Learning for Recommender Systems.CoRRabs/1606.07792(2016). 
arXiv:1606.07792 http://arxiv.org/abs/1606.07792 ).  
[Original Data] (https://github.com/fengtong-xiao/DMBGN/tree/master/data).
