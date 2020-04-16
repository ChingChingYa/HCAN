import pandas as pd
import torch
import numpy as np
from numpy.random import RandomState
import copy
from load_data import load_data, read_dataset, cut_data_len
from sklearn.metrics import mean_squared_error
from sklearn import metrics
from sklearn.model_selection import train_test_split

# https://github.com/mcleonard/pmf-pytorch/blob/master/Probability%20Matrix%20Factorization.ipynb


class PMF(torch.nn.Module):
    def __init__(self, U, V, lambda_U=1e-2, lambda_V=1e-2, latent_size=5,
                 momentum=0.8, learning_rate=0.001, iterations=1000):
        super().__init__()
        self.lambda_U = lambda_U
        self.lambda_V = lambda_V
        # momentum動量  避免曲折過於嚴重
        self.momentum = momentum
        # k數量
        self.latent_size = latent_size
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.U = None
        self.V = None
        self.R = None
        self.I = None
        self.count_users = None
        self.count_items = None


    def RMSE(self, predicts, truth):
        # print(self.count_users)
        # return np.sqrt(np.mean(np.square(predicts - truth)))
        return torch.sqrt(mean_squared_error(predicts, truth))

    # 原始矩陣分解法目標式
    def loss(self, aspect):
        # the loss function of the model
        if(aspect == []):
            loss = (torch.sum(self.I*(self.R-torch.mm(self.U, self.V.t()))**2)
                + self.lambda_U*torch.sum(self.U.pow(2))
                + self.lambda_V*torch.sum(self.V.pow(2)))
        else:
            loss = (torch.sum(self.I*(self.R-torch.mm(self.U, self.V.t()))**2)
                    + self.lambda_U*torch.sum(self.U.pow(2))
                    + self.lambda_V*torch.sum(self.V.pow(2))+(torch.mul(self.U, self.V)-aspect)**2)
        return loss


    def predict(self, data):
        index_data = torch.IntTensor([[int(ele[0]), int(ele[1])] for ele in data])
        torch.take(x, torch.LongTensor(indices))
        u_features = self.U.take(index_data.take(0, axis=1), axis=0)
        v_features = self.V.take(index_data.take(1, axis=1), axis=0)
        a = u_features*v_features
        preds_value_array = torch.sum(u_features*v_features, 1)
        return preds_value_array    


    def forward(self, num_users, num_items, train_data=None,
                aspect_vec=None, U=None, V=None, flag=0,
                lambda_U=0.01, lambda_V=0.01):


        self.lambda_U = lambda_U
        self.lambda_V = lambda_V

        aspect = []
        tmp = np.array([0]*6, dtype=np.float32)

        if flag != 0:
            x = int(len(aspect_vec)/0.8*64*0.2)
            for i in range(x):
                aspect.append(tmp)
            for i, a in enumerate(aspect_vec):
                for j in range(len(a)):
                    aspect.append(aspect_vec[i][j])

        if self.R is None:
            self.R = torch.zeros((num_users, num_items))
            for i, ele in enumerate(train_data):
                self.R[int(ele[0]), int(ele[1])] = float(ele[2])
            # 有評分為1 為評分為0
            self.I = copy.deepcopy(self.R)
            self.I[self.I != 0] = 1

        if self.U is None and self.V is None:
            self.count_users = np.size(self.R, 0)
            self.count_items = np.size(self.R, 1)
            # Random
            self.U = torch.randn(self.count_users, self.latent_size)
            self.U.requires_grad = True

            self.V = torch.randn(self.count_items, self.latent_size)
            self.V.requires_grad = True
        else:
            self.U = U
            self.V = V   

        optimizer = torch.optim.SGD([self.U, self.V], lr=self.learning_rate, momentum=self.momentum)

        loss_list = []

        for step, epoch in enumerate(range(self.iterations)):
            optimizer.zero_grad()
            loss = self.loss(aspect)
            loss_list.append(loss.detach().numpy())
            loss.backward()
            optimizer.step()
            if step % 50 == 0:
                print(f"Step {step}, {loss:.3f}")

        u_list = [i.detach().numpy() for i in self.U]
        v_list = [i.detach().numpy() for i in self.V]
        loss_list = np.sum(loss_list)

        return u_list, v_list, loss_list/len(train_data)            
