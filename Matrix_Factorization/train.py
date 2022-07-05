import yaml
from data import dataset_model_svd , dataset_model_torch
from model_svd import SVD
import torch
from model_torch import MF_torch
from tqdm import tqdm

#Reading configuration yaml file for hyperparamters
with open('configuration.yaml') as f:
    config = yaml.safe_load(f)
#argparse 또는 yaml로 model 선택하는 부분 공부 진행 중...!

#==============================When using model_svd==================================

train_data, test_data = dataset_model_svd()
model_svd = SVD(train_data,test_data,config['k'],config['learning_rate'],config['cost_parameter'],config['epochs'])
model_svd.train()

#==============================When using model_torch================================
class Trainer(object):
    def __init__(self, dataset, model,config):
        self.dataset = dataset
        self.model = model
        self.config = config
        self.device = self._get_device()
        self.loss = torch.nn.MSELoss()
        self.model_dict = {'model_pytorch':MF_torch}
        self.train = self.train()


    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'running on {device}')
        return device

    def train(self):

        # model = self._get_model()
        # model = model(**self.config['model'])

        model = self.model.to(self.device)
        model.train()

        criterion = self.loss.to(self.device)
        optimizer = torch.optim.SGD(model.parameters(),lr=self.config['learning_rate'])
        epochs = self.config['epochs']

        rows, cols = self.dataset.nonzero()
        for epoch in tqdm(range(epochs),colour='BLUE'):
            train_loss = 0
            for row, col in zip(*(rows, cols)):
                optimizer.zero_grad()
                rating = torch.FloatTensor([self.dataset[row,col]])
                row = torch.LongTensor([row])
                col = torch.LongTensor([col])

                prediction = model(row, col)
                loss = criterion(prediction, rating)
                train_loss += loss.item()

                loss.backward()
                optimizer.step()

            print(f'Epoch {epoch+1}/{epochs} // Loss:{train_loss/len(rows)}')

Trainer(dataset_model_torch(),MF_torch(dataset_model_torch(),610,9724,20),config)
#==============================When using model_als==================================



















'''

#model selection
    def _get_model(self):
        try:
            model = self.model_dict[self.model]
        except:
            raise(f'Wrong name, give proper name from dict')
'''

