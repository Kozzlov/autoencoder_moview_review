import numpy as np 
import pandas as pd 

import torch 
import torch.nn as nn 
import torch.nn.parallel 
import torch.optim as optim 
import torch.utils.data 
from torch.autograd import Variable 

movies = pd.read_csv('ml-1m/movies.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1') 
users = pd.read_csv('ml-1m/users.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1') 
ratings = pd.read_csv('ml-1m/ratings.dat', sep = '::', header = None, engine = 'python', encoding = 'latin-1')


training_set = pd.read_csv('ml-100k/u1.base', delimiter = '\t')
training_set = np.array(training_set, dtype = 'int')
test_set = pd.read_csv('ml-100k/u1.test', delimiter = '\t')
test_set = np.array(test_set, dtype = 'int')

nbr_u = int(max(max(training_set[:, 0]), max(test_set[:, 0])))
nbr_m = int(max(max(training_set[:, 1]), max(test_set[:, 1])))

def convert(data):
    new_data = []
    for user_id in range(1, nbr_u + 1):
        movies_id = data[:, 1][data[:, 0] == user_id]
        ratings_id = data[:, 2][data[:, 0] == user_id]
        ratings = np.zeros(nbr_m)
        ratings[movies_id - 1] = ratings_id
        new_data.append(list(ratings))
    return new_data
training_set = convert(training_set)
test_set = convert(test_set)
 
training_set = torch.FloatTensor(training_set)
test_set = torch.FloatTensor(test_set)

class SAE (nn.Module):
    def __init__(self, ):
        super(SAE, self).__init__()
        self.fc1 = nn.Linear(nbr_m, 20)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, 20)
        self.fc4 = nn.Linear(20, nbr_m)
        self.activation = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.activation(self.fc3(x))
        x = self.fc4(x)
        return x


sae = SAE() 
criterion = nn.MSELoss() 
optimizer = optim.RMSprop(sae.parameters(), lr = 0.01, weight_decay = 0.5)


nb_epoch = 150

for epoch in range(1, nb_epoch + 1):
    train_loss = 0
    s = 0.
    for user_id in range(nbr_u):
        input = Variable(training_set[user_id]).unsqueeze(0) 
        target = input.clone()
        if torch.sum(target.data > 0) >  0:
            output = sae(input)
            target.require_grad = False
            output[target == 0] = 0
            loss = criterion(output, target)
            mean_corrector = nbr_m/float(torch.sum(target.data > 0) + 1e-10)
            loss.backward()
            train_loss += np.sqrt(loss.data*mean_corrector)
            s += 1.
            optimizer.step()
    print(' epoch: '+str(epoch)+' test loss: '+str(train_loss/s))

test_loss = 0
s = 0.
for user_id in range(nbr_u):
    input = Variable(training_set[user_id]).unsqueeze(0) 
    target = Variable(test_set[user_id]).unsqueeze(0) 
    if torch.sum(target.data > 0) >  0:
        output = sae(input)
        target.require_grad = False
        output[target == 0] = 0
        loss = criterion(output, target)
        mean_corrector = nbr_m/float(torch.sum(target.data > 0) + 1e-10)
        test_loss += np.sqrt(loss.data*mean_corrector)
        s += 1.
print('train loss: '+str(test_loss/s))
