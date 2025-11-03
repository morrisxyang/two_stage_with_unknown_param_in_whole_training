import os
import sys
import create_folder as create_folder
import numpy as np
import random
import pandas as pd
import math, time
import itertools
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import datetime
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.utils.data as data_utils
from torch.utils.data.dataset import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import gurobipy as gp
import logging
import copy
from collections import defaultdict
import joblib
import gurobipy as gp
from gurobipy import GRB

capacity = 250
purchase_fee = 1
compensation_fee = 10

item_num = 10
feature_num = 4096
train_case_num = 700
target_num = 2
ReLUValue = 25

prev_folder_path = os.path.abspath(os.path.dirname(os.getcwd()))
npy_data_path = os.path.join(prev_folder_path, 'data/train_size=' + str(train_case_num) + '/npy_version/')
txt_data_path = os.path.join(prev_folder_path, 'data/train_size=' + str(train_case_num) + '/txt_version/')
#res_path = os.path.join(prev_folder_path, 'data/train_size=' + str(train_case_num) + '/')
create_folder.mkdir(os.path.join(prev_folder_path, 'data/train_size=' + str(train_case_num) + '/'), 'cap=' + str(capacity) + ',K=' + str(compensation_fee))
create_folder.mkdir(os.path.join(prev_folder_path, 'data/train_size=' + str(train_case_num) + '/cap=' + str(capacity) + ',K=' + str(compensation_fee) + '/'), 'proposed_prices')
create_folder.mkdir(os.path.join(prev_folder_path, 'data/train_size=' + str(train_case_num) + '/cap=' + str(capacity) + ',K=' + str(compensation_fee) + '/'), 'proposed_weights/')
store_price_path = os.path.join(prev_folder_path, 'data/train_size=' + str(train_case_num) + '/cap=' + str(capacity) + ',K=' + str(compensation_fee) + '/proposed_prices')
store_weight_path = os.path.join(prev_folder_path, 'data/train_size=' + str(train_case_num) + '/cap=' + str(capacity) + ',K=' + str(compensation_fee) + '/proposed_weights/')


def actual_obj(valueTemp, cap, weightTemp, n_instance):
    obj_list = []
    selectedNum_list = []
    for num in range(n_instance):
        weight = np.zeros(item_num)
        value = np.zeros(item_num)
        cnt = num * item_num
        for i in range(item_num):
            weight[i] = weightTemp[cnt]
            value[i] = valueTemp[cnt]
            cnt = cnt + 1
        weight = weight.tolist()
        value = value.tolist()
        
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(item_num, vtype=GRB.BINARY, name='x')
        m.setObjective(purchase_fee * x.prod(value), GRB.MAXIMIZE)
        m.addConstr((x.prod(weight)) <= cap)
#        for i in range(item_num):
#            m.addConstr((x.prod(weight[i])) <= cap)

        m.optimize()
        objective = m.objVal
        obj_list.append(objective)
        
#        sol = []
#        selecteditem_num = 0
#        for i in range(item_num):
#            sol.append(x[i].x)
#            if x[i].x == 1:
#              selecteditem_num = selecteditem_num + 1
#        selectedNum_list.append(selecteditem_num)
        # print(selecteditem_num)
#        print("TOV: ", sol, objective)
        
    return np.array(obj_list)


def get_true_x_s1(pred_cost_temp, pred_weight_temp, cap):

    pred_cost = pred_cost_temp.tolist()
    pred_weight = pred_weight_temp.tolist()
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    x = m.addVars(item_num, vtype=GRB.BINARY, name='x')
    m.setObjective(purchase_fee * x.prod(pred_cost), GRB.MAXIMIZE)
    m.addConstr((x.prod(pred_weight)) <= cap)

    m.optimize()
    predSol = np.zeros(item_num,dtype='i')
#        x1_selecteditem_num = 0
    for i in range(item_num):
        predSol[i] = x[i].x
#            if x[i].x == 1:
#              x1_selecteditem_num = x1_selecteditem_num + 1
    objective1 = m.objVal
    

    return predSol
    


def correction_single_obj(realPrice, predPrice, cap, realWeightTemp, predWeightTemp):
#    print("realPrice: ", realPrice, "predPrice: ", predPrice)
    realWeight = np.zeros(item_num)
    predWeight = np.zeros(item_num)
    realPriceNumpy = np.zeros(item_num)
    for i in range(item_num):
        realWeight[i] = realWeightTemp[i]
        predWeight[i] = predWeightTemp[i]
        realPriceNumpy[i] = realPrice[i]
        
    if min(predWeight) >= 0:
        predWeight = predWeight.tolist()
        m = gp.Model()
        m.setParam('OutputFlag', 0)
        x = m.addVars(item_num, vtype=GRB.BINARY, name='x')
        m.setObjective(purchase_fee * x.prod(predPrice), GRB.MAXIMIZE)
        m.addConstr((x.prod(predWeight)) <= cap)

        m.optimize()
        predSol = np.zeros(item_num,dtype='i')
#        x1_selecteditem_num = 0
        for i in range(item_num):
            predSol[i] = x[i].x
#            if x[i].x == 1:
#              x1_selecteditem_num = x1_selecteditem_num + 1
        objective1 = m.objVal
#        print("Stage 1: ", predSol, objective1)

        # Stage 2:
        realWeight = realWeight.tolist()
        m2 = gp.Model()
        m2.setParam('OutputFlag', 0)
        x1 = m2.addVars(item_num, vtype=GRB.BINARY, name='x1')
        x2 = m2.addVars(item_num, vtype=GRB.BINARY, name='x2')
        sigma = m2.addVars(item_num, vtype=GRB.BINARY, name='sigma')

        OBJ = purchase_fee * x2.prod(realPrice)
        for i in range(item_num):
            OBJ = OBJ - compensation_fee * sigma[i]
        m2.setObjective(OBJ, GRB.MAXIMIZE)

        m2.addConstr((x2.prod(realWeight)) <= cap)
        for i in range(item_num):
            m2.addConstr(x1[i] == predSol[i])
            m2.addConstr(sigma[i] >= x2[i] - x1[i])
            m2.addConstr(sigma[i] >= x1[i] - x2[i])
        
        try:
            m2.optimize()
            objective = m2.objVal
#            sol = []
#            x2_selecteditem_num = 0
#            for i in range(item_num):
#                sol.append(x[i].x - sigma[i].x)
#                if x[i].x - sigma[i].x == 1:
#                  x2_selecteditem_num = x2_selecteditem_num + 1
    #        print("Stage 2: ", sol, objective)
        except:
            print(predPrice, predWeight, realPrice, realWeight, predSol)

    return objective
    
# simply define a silu function
def silu(input):
    for i in range(item_num):
        if input[i][0] < 0:
            input[i][0] = 0
        input[i][0] = input[i][0] + ReLUValue
        if input[i][1] < 0:
            input[i][1] = 0
        input[i][1] = input[i][1] + ReLUValue
    return input

# create a class wrapper from PyTorch nn.Module, so
# the function now can be easily used in models
class SiLU(nn.Module):
    def __init__(self):
        super().__init__() # init the base class

    def forward(self, input):
        return silu(input) # simply apply already implemented SiLU

# initialize activation function
activation_function = SiLU()

    
def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)

    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


def make_fc(num_layers, num_features, num_targets=target_num,
            activation_fn = nn.ReLU,intermediate_size=512, regularizers = True):
    net_layers = [nn.Linear(num_features, intermediate_size),activation_fn()]
    for hidden in range(num_layers-2):
        net_layers.append(nn.Linear(intermediate_size, intermediate_size))
        net_layers.append(activation_fn())
    net_layers.append(nn.Linear(intermediate_size, num_targets))
    net_layers.append(activation_function)
    return nn.Sequential(*net_layers)
        

class MyCustomDataset():
    def __init__(self, feature, value):
        self.feature = feature
        self.value = value

    def __len__(self):
        return len(self.value)

    def __getitem__(self, idx):
        return self.feature[idx], self.value[idx]


import sys
import ip_model_whole as ip_model_wholeFile
from ip_model_whole import IPOfunc

class Intopt:
    def __init__(self, opt_sol, h, A, b, purchase_fee, compensation_fee, n_features, num_layers=5, smoothing=False, thr=0.1, max_iter=None, method=1, mu0=None,
        damping=0.5, target_size=target_num, epochs=8, optimizer=optim.Adam,
        batch_size=item_num, **hyperparams):
        
        self.opt_sol = opt_sol
        self.h = h
        self.A = A
        self.b = b
        self.target_size = target_size
        self.n_features = n_features
        self.damping = damping
        self.num_layers = num_layers
        self.purchase_fee = purchase_fee
        self.compensation_fee = compensation_fee

        self.smoothing = smoothing
        self.thr = thr
        self.max_iter = max_iter
        self.method = method
        self.mu0 = mu0

        self.optimizer = optimizer
        self.batch_size = batch_size
        self.hyperparams = hyperparams
        self.epochs = epochs
        # print("embedding size {} n_features {}".format(embedding_size, n_features))

#        self.model = Net(n_features=n_features, target_size=target_size)
        self.model = make_fc(num_layers=self.num_layers,num_features=n_features)
        #self.model.apply(weight_init)
#        w1 = self.model[0].weight
#        print(w1)

        self.optimizer = optimizer(self.model.parameters(), **hyperparams)

    def fit(self, feature, value):
        logging.info("Intopt")
        train_df = MyCustomDataset(feature, value)

        criterion = nn.L1Loss(reduction='mean')  # nn.MSELoss(reduction='mean')
        grad_list = np.zeros(self.epochs)
        for e in range(self.epochs):
          total_loss = 0
#          for parameters in self.model.parameters():
#            print(parameters)
          if e < 0:
            #print('stage 1')
            train_dl = data_utils.DataLoader(train_df, batch_size=self.batch_size, shuffle=False)
            for feature, value in train_dl:
                self.optimizer.zero_grad()
                op = self.model(feature).squeeze()
#                print(feature, value, op)
#                print(feature.shape, value.shape, op.shape)
                # target_num=1: torch.Size([10, 4096]) torch.Size([10]) torch.Size([10])
                # target_num=2: torch.Size([10, 4096]) torch.Size([10, 2]) torch.Size([10, 2])
#                print(value, op)
                
                loss = criterion(op, value)
                total_loss += loss.item()
                loss.backward()
                self.optimizer.step()
            grad_list[e] = total_loss
            print("Epoch{} ::loss {} ->".format(e,total_loss))
                
          else:
#            if e == 1:
#                for param_group in self.optimizer.param_groups:
#                    param_group['lr'] = 1e-5
            #print('stage 2')
            train_dl = data_utils.DataLoader(train_df, batch_size=self.batch_size, shuffle=False)
            
            num = 0
            batchCnt = 0
            loss = Variable(torch.tensor(0.0, dtype=torch.double), requires_grad=True)
            for feature, value in train_dl:
                cur_opt_sol = self.opt_sol[batchCnt]
                cur_opt_sol = torch.from_numpy(cur_opt_sol).float()

                self.optimizer.zero_grad()
                op = self.model(feature).squeeze()
                while torch.min(op) <= 0 or torch.isnan(op).any() or torch.isinf(op).any():
                    self.optimizer.zero_grad()
#                    self.model.__init__(self.n_features, self.target_size)
                    self.model = make_fc(num_layers=self.num_layers,num_features=self.n_features)
                    op = self.model(feature).squeeze()
  
#                price = np.zeros(item_num)
#                for i in range(item_num):
#                    price[i] = self.c[i+num*item_num]
#                    op[i] = op[i]
                
                
#                c_torch = torch.from_numpy(price).float()
                h_torch = torch.from_numpy(self.h).float()
                A_torch = torch.from_numpy(self.A).float()
                b_torch = torch.from_numpy(self.b).float()
                
                c_torch = value[:, 0]
                G_torch = torch.zeros((item_num+1, item_num))
                for i in range(item_num):
                    G_torch[i][i] = 1
                G_torch[item_num] = value[:, 1]
                trueWeight = value[:, 1]
                

                x_s1 = IPOfunc(cur_opt=cur_opt_sol,A=A_torch, b=b_torch, h=h_torch, cTrue=-c_torch, GTrue=G_torch, purchase_fee=self.purchase_fee, compensation_fee=self.compensation_fee, max_iter=self.max_iter, thr=self.thr, damping=self.damping,
                            smoothing=self.smoothing)(op)
                #print(c_torch.shape, G_torch.shape, x.shape)    # torch.Size([242]) torch.Size([43, 242]) torch.Size([242])
#                x_s1 = ip_model_wholeFile.x_s1
#                print(x_s1, cur_opt_sol)
                newLoss = compensation_fee * abs(x_s1 - cur_opt_sol).sum()
                
#                pred_cost = op[:, 0]
#                pred_weight = op[:, 1]
#                pred_cost = pred_cost.detach().numpy()
#                pred_weight = pred_weight.detach().numpy()
#                x_s1_IP = get_true_x_s1(pred_cost, pred_weight, capacity)
#                x_s1_IP = torch.from_numpy(x_s1_IP).float()
##                print(x_s1_IP, cur_opt_sol, abs(x_s1_IP - cur_opt_sol))
#                newLoss.data = compensation_fee * abs(x_s1_IP - cur_opt_sol).sum()

#                print(newLoss)
#                time.sleep(100)
#                newLoss = - (purchase_fee * (x_s2 * c_torch).sum() - (compensation_fee - purchase_fee) * torch.dot(c_torch, abs(x_s2-x_s1).float()))

                loss = loss + newLoss
                batchCnt = batchCnt + 1

                total_loss += newLoss.item()
                # op.retain_grad()
                #print(loss)
                
                newLoss.backward()
                #print("backward1")
                self.optimizer.step()
                
                # when training size is large
                if batchCnt % 350 == 0:
                    print(newLoss)
#                    newLoss.backward()
#                    #print("backward1")
#                    self.optimizer.step()
                num = num + 1
            grad_list[e] = total_loss/train_case_num
            print("Epoch{} ::loss {} ->".format(e,grad_list[e]))
                
          
          logging.info("EPOCH Ends")
          #print("Epoch{}".format(e))
#          for param_group in self.optimizer.param_groups:
#            print(param_group['lr'])
          if e > 0 and abs(grad_list[e] - grad_list[e-1]) <= 0.01:
            break
#          if grad_list[e] < stopCriterior:
#            break
#          if e > 0 and grad_list[e] >= grad_list[e-1]:
#            break
          if grad_list[e] > 10000:
            break
#          else:
#            currentBestLoss = total_loss
#          if total_loss > -500:
#            break
#           print(self.val_loss(valid_econ, valid_prop))
          # print("______________")

    def val_loss(self, cap, feature, value):
        valueTemp = value.numpy()
#        test_instance = len(valueTemp) / self.batch_size
        test_instance = np.size(valueTemp, 0) / self.batch_size
#        itemVal = self.c.tolist()
#        itemVal = self.c
        itemVal = valueTemp[:, 0]
        real_obj = actual_obj(itemVal, cap, value[:, 1], n_instance=int(test_instance))
#        print(np.sum(real_obj))

        self.model.eval()
        criterion = nn.L1Loss(reduction='mean')  # nn.MSELoss(reduction='sum')
        valid_df = MyCustomDataset(feature, value)
        valid_dl = data_utils.DataLoader(valid_df, batch_size=self.batch_size, shuffle=False)

        obj_list = []
        corr_obj_list = []
        len = np.size(valueTemp, 0)
        predVal = torch.zeros((len, 2))
        
        num = 0
        for feature, value in valid_dl:
            op = self.model(feature).squeeze()
#            print(op)
            loss = criterion(op, value)

            realWT = {}
            predWT = {}
            realPrice = {}
            predPrice = {}
            for i in range(item_num):
                realWT[i] = value[i][1]
                predWT[i] = op[i][1]
                realPrice[i] = value[i][0]
                predPrice[i] = op[i][0]
                predVal[i+num*item_num][0] = op[i][0]
                predVal[i+num*item_num][1] = op[i][1]

            corrrlst = correction_single_obj(realPrice, predPrice, cap, realWT, predWT)
            corr_obj_list.append(corrrlst)
            num = num + 1
            

        self.model.train()
#        print(corr_obj_list)
#        print(corr_obj_list-real_obj)
#        print(np.sum(corr_obj_list))
#        return prediction_loss, abs(np.array(obj_list) - real_obj)
        return abs(np.array(corr_obj_list) - real_obj), predVal


#c_dataTemp = np.loadtxt('KS_c.txt')
#c_data = c_dataTemp[:item_num]

h_data = np.ones(item_num+1)
h_data[item_num] = capacity
A_data = np.zeros((2, item_num))
b_data = np.zeros(2)


#startmark = int(sys.argv[1])
#endmark = startmark + 30

print('capacity: ', capacity, 'purchase_fee: ', purchase_fee, 'compensation_fee: ', compensation_fee)
print("*** Pen as loss ****")

#for testmark in range(startmark, endmark):
    #recordFile = open('record(' + str(testmark) + ').txt', 'a')
testTime = 10
#test_list = [2,4,5,6,7,8,9]
recordBest = np.zeros((1, testTime))

for testi in range(testTime):
#for testi in test_list:
    print(testi)
    train_opt_sol = np.loadtxt(os.path.join(txt_data_path, 'train_optimal_sol/cap=' + str(capacity) + '/train_optimal_sol(' + str(testi) + ').txt'))
    x_train = np.loadtxt(os.path.join(txt_data_path, 'train_features/train_features(' + str(testi) + ').txt'))
    y_train1 = np.loadtxt(os.path.join(txt_data_path, 'train_prices/train_prices(' + str(testi) + ').txt'))
    y_train2 = np.loadtxt(os.path.join(txt_data_path, 'train_weights/train_weights(' + str(testi) + ').txt'))
#    penalty_train = np.loadtxt('./CombOptNet/train_case_num700/train_penalty' + str(penaltyTerm) + '/train_penalty(' + str(testi) + ').txt')


    y_train = np.zeros((y_train1.size, 2))
    for i in range(y_train1.size):
        y_train[i][0] = y_train1[i]
        y_train[i][1] = y_train2[i]
    feature_train = torch.from_numpy(x_train).float()
    value_train = torch.from_numpy(y_train).float()
    
    test_opt_sol = np.loadtxt(os.path.join(txt_data_path, 'test_optimal_sol/cap=' + str(capacity) + '/test_optimal_sol(' + str(testi) + ').txt'))
    x_test = np.loadtxt(os.path.join(txt_data_path, 'test_features/test_features(' + str(testi) + ').txt'))
    y_test1 = np.loadtxt(os.path.join(txt_data_path, 'test_prices/test_prices(' + str(testi) + ').txt'))
    y_test2 = np.loadtxt(os.path.join(txt_data_path, 'test_weights/test_weights(' + str(testi) + ').txt'))
#    penalty_test = np.loadtxt('./CombOptNet/train_case_num700/test_penalty' + str(penaltyTerm) + '/test_penalty(' + str(testi) + ').txt')

    y_test = np.zeros((y_test1.size, 2))
    for i in range(y_test1.size):
        y_test[i][0] = y_test1[i]
        y_test[i][1] = y_test2[i]
    feature_test = torch.from_numpy(x_test).float()
    value_test = torch.from_numpy(y_test).float()
    
    start = time.time()
    damping = 1e-2
    thr = 1e-3
    lr = 1e-3
#    lr = 1e-5
    if compensation_fee == 5:
        stopCriterior = 25
    elif compensation_fee == 10:
        stopCriterior = 50
    elif compensation_fee == 20:
        stopCriterior = 60
    bestTrainCorrReg = float("inf")
    for j in range(5):
        clf = Intopt(train_opt_sol, h_data, A_data, b_data, purchase_fee, compensation_fee, damping=damping, lr=lr, n_features=feature_num, thr=thr, epochs=4)
        clf.fit(feature_train, value_train)
        train_rslt, predTrainVal = clf.val_loss(capacity, feature_train, value_train)
        avgTrainCorrReg = np.mean(train_rslt)
    #    trainHSD_rslt = str(testmark) + ' train: ' + str(np.sum(train_rslt[1])) + ' ' + str(np.mean(train_rslt[1]))
        trainHSD_rslt = 'train: ' + str(np.mean(train_rslt))

        if avgTrainCorrReg < bestTrainCorrReg:
            bestTrainCorrReg = avgTrainCorrReg
            torch.save(clf.model.state_dict(), 'model.pkl')
        print(trainHSD_rslt)
        
        if avgTrainCorrReg < stopCriterior:
            break

#        val_rslt = clf.val_loss(source, sink, arc, feature_test, value_test)
#        #HSD_rslt = str(testmark) + ' test: ' + str(np.sum(val_rslt[0])) + ' ' + str(np.sum(val_rslt[1]))
#        HSD_rslt = ' test: ' + str(np.sum(val_rslt[0])) + ' ' + str(np.sum(val_rslt[1]))
#        print(HSD_rslt)

#    val_rslt = clf.val_loss(source, sink, arc, feature_test, value_test)
##    HSD_rslt = str(testmark) + ' test: ' + str(np.sum(val_rslt[0])) + ' ' + str(np.sum(val_rslt[1]))
#    HSD_rslt = ' test: ' + str(np.sum(val_rslt[0])) + ' ' + str(np.sum(val_rslt[1]))
#    print(HSD_rslt)
#    print('\n')
#    recordBest[0][i] = np.sum(val_rslt[1])

    clfBest = Intopt(test_opt_sol, h_data, A_data, b_data, purchase_fee, compensation_fee, damping=damping, lr=lr, n_features=feature_num, thr=thr, epochs=8)
    clfBest.model.load_state_dict(torch.load('model.pkl'))

    val_rslt, predTestVal = clfBest.val_loss(capacity, feature_test, value_test)
    #HSD_rslt = str(testmark) + ' test: ' + str(np.sum(val_rslt[0])) + ' ' + str(np.sum(val_rslt[1]))
#    print(predTestVal.shape)
    end = time.time()

    predTestVal = predTestVal.detach().numpy()
#    print(predTestVal.shape)
    predTestVal1 = predTestVal[:, 0]
    predTestVal2 = predTestVal[:, 1]
    predValuePrice = np.zeros((predTestVal1.size, 2))
    for i in range(predTestVal1.size):
#        predValue[i][0] = int(i/item_num)
        predValuePrice[i][0] = y_test1[i]
        predValuePrice[i][1] = predTestVal1[i]
            
    np.savetxt(os.path.join(store_price_path, 'proposed_prices(' + str(testi) + ').txt'), predValuePrice, fmt="%.2f")
    predValueWeight = np.zeros((predTestVal2.size, 2))
    for i in range(predTestVal2.size):
#        predValue[i][0] = int(i/item_num)
        predValueWeight[i][0] = y_test2[i]
        predValueWeight[i][1] = predTestVal2[i]
    np.savetxt(os.path.join(store_weight_path, 'proposed_weights(' + str(testi) + ').txt'), predValueWeight, fmt="%.2f")
    
    HSD_rslt = 'test: ' + str(np.mean(val_rslt))
    print(HSD_rslt)
    print ('Elapsed time: ' + str(end-start))
    recordBest[0][testi] = np.sum(val_rslt)

print(recordBest)
