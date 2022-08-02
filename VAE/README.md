# VAE  
Auto-Encoding Variational Bayes (Diederik P.Kingma, Max Welling)  
Paper proposed a generative model even works efficiently in the case of "Intractability" and "Large Dataset".  
Using variational inference, finding 'phi' that maximizes ELBO instead of minimizing KL term.



## Model  
### Overal structure of VAE  
![architecture_vae](https://user-images.githubusercontent.com/62690984/182337285-b868876d-625b-420d-a470-5993583f6dee.png)  
### Variational lower bound with KL term, Gaussian case  
![kl_divergence_error](https://user-images.githubusercontent.com/62690984/182335350-f08be8d0-a6e8-40f5-8aa0-56dc488c0323.png)  
### Bernoulli MLP as decoder  
![captured_bce](https://user-images.githubusercontent.com/62690984/182336371-5ac08068-7854-4682-ba38-d1cd53f45ebc.png)  
### Gaussian MLP as encoder or decoder  
![captured_gaussian_decoder](https://user-images.githubusercontent.com/62690984/182336474-e327c3b5-732c-419f-afed-0fda81fa0d7c.png)  
## Usage  
For using decoder as Bernoulli decoder, you have to use 'sigmoid_output' in model.py and using 'loss_function' with BCE in train.py.    
For using decoder as Gaussian decoder, you have to use just 'output' (not passed by sigmoid) in model.py and using 'loss_function' with MSE loss in train.py.  
There are pretrained models' weight in checkpoints files  
## Results  
### Mnist using Bernoulli decoder   
(First row: original image, Second row: generated image)  
![bce_vae_results_captured](https://user-images.githubusercontent.com/62690984/182333540-4cd6474e-2b63-4985-81e1-4d6ef88e116f.png)  
### Mnist using Gaussian decoder  
(First row: original image, Second row: generated image)    
![mse_vae_results_captured](https://user-images.githubusercontent.com/62690984/182333609-7eeda44b-e489-4355-9fb9-f18ad735b75a.png)



## References  
[1] "Auto-encoding variational bayes",https://arxiv.org/abs/1312.6114  
[2] Overal structure image from hwalsuklee's slides  
