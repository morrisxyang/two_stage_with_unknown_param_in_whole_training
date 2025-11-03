import os
import sys
from collections import defaultdict
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from sklearn.metrics import mean_squared_error
from numpy import inf

purchase_fee = 1
compensation_fee = 20
cap_list = [100, 150, 200, 250]
train_case_num = 700
test_case_num = 1
item_num = 10
#startmark = int(sys.argv[1])
startmark = 0
endmark = startmark + 1
ReLUVal = 25

prev_folder_path = os.path.abspath(os.path.dirname(os.getcwd()))


# Create an environment with your WLS license
params = {
"WLSACCESSID": '32387500-f923-4275-8cb4-4a0fd23e1e88',
"WLSSECRET": '5e17ec44-0c01-4974-b1bb-ac737669e41c',
"LICENSEID": 2598743,
}

env = gp.Env(params=params)

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
        sol = []
        selecteditem_num = 0
        for i in range(item_num):
            sol.append(x[i].x)
            if x[i].x == 1:
              selecteditem_num = selecteditem_num + 1
        objective = m.objVal
        obj_list.append(objective)
        selectedNum_list.append(selecteditem_num)
        # print(selecteditem_num)
#        print("TOV: ", sol, objective)
        
    return np.array(obj_list), np.array(selectedNum_list)
    
    
def compute_training_loss(realPrice, realWeight, predPrice, predWeight, cap):
    real_price = realPrice.tolist()
    real_weight = realWeight.tolist()
    pred_weight = predWeight.tolist()
    pred_price = predPrice.tolist()
    
    # compute true opt sol
    m_true = gp.Model()
    m_true.setParam('OutputFlag', 0)
    x = m_true.addVars(item_num, vtype=GRB.BINARY, name='x')
    m_true.setObjective(purchase_fee * x.prod(real_price), GRB.MAXIMIZE)
    m_true.addConstr((x.prod(real_weight)) <= cap)
    m_true.optimize()
    opt_sol = np.zeros(item_num)
    for i in range(item_num):
        opt_sol[i] = x[i].x
    
    # compute pred opt sol
    m_pred = gp.Model()
    m_pred.setParam('OutputFlag', 0)
    x_pred = m_pred.addVars(item_num, vtype=GRB.BINARY, name='x')
    m_pred.setObjective(purchase_fee * x_pred.prod(pred_price), GRB.MAXIMIZE)
    m_pred.addConstr((x_pred.prod(pred_weight)) <= cap)
    m_pred.optimize()
    pred_sol = np.zeros(item_num)
    for i in range(item_num):
        pred_sol[i] = x_pred[i].x
    
    print("true opt sol: ", opt_sol)
    print("pred opt sol: ", pred_sol)
    
    loss = compensation_fee * abs(pred_sol - opt_sol).sum()
    
    return loss
    
    
    
def compute_single_correction_obj(realPrice, realWeight, predPrice, predWeight, cap):
    pred_weight = predWeight.tolist()
    pred_price = predPrice.tolist()
    
    m = gp.Model()
    m.setParam('OutputFlag', 0)
    x = m.addVars(item_num, vtype=GRB.BINARY, name='x')
    m.setObjective(purchase_fee * x.prod(pred_price), GRB.MAXIMIZE)
    m.addConstr((x.prod(pred_weight)) <= cap)
#        for i in range(item_num):
#            m.addConstr((x.prod(weight[i])) <= cap)

    m.optimize()
    predSol = np.zeros(item_num)
    for i in range(item_num):
        predSol[i] = x[i].x
#    print("pred opt sol: ", predSol)
    
    # Stage 2:
    realPrice = realPrice.tolist()
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

    m2.optimize()
    objective = m2.objVal
    final_sol = np.zeros(item_num)
    for i in range(item_num):
        final_sol[i] = x2[i].x
    print("corr opt sol: ", final_sol)
    
#    x1_sol = np.zeros(item_num)
#    x2_sol = np.zeros(item_num)
#    sigma_sol = np.zeros(item_num)
#    for i in range(item_num):
#        x1_sol[i] = x1[i].x
#        x2_sol[i] = x2[i].x
#        sigma_sol[i] = sigma[i].x
#    print("x1: ", x1_sol, "x2: ", x2_sol, "sigma: ", sigma_sol)
    
#    sol = []
#    x2_selecteditem_num = 0
#    for i in range(item_num):
#        sol.append(x[i].x - sigma[i].x)
#        if x[i].x - sigma[i].x == 1:
#          x2_selecteditem_num = x2_selecteditem_num + 1
#        print("Stage 2: ", sol, objective)

    return objective



for capacity in cap_list:
    price_path = os.path.join(prev_folder_path, 'data/train_size=' + str(train_case_num) + '/cap=' + str(capacity) + ',K=' + str(compensation_fee) + '/proposed_prices')
    weight_path = os.path.join(prev_folder_path, 'data/train_size=' + str(train_case_num) + '/cap=' + str(capacity) + ',K=' + str(compensation_fee) + '/proposed_weights/')
    print('capacity: ', capacity, 'purchase_fee: ', purchase_fee, 'compensation_fee: ', compensation_fee)
    
    
    for testmark in range(startmark, endmark):
        price_file = np.loadtxt(os.path.join(price_path, 'proposed_prices(' + str(testmark) + ').txt'))
        weight_file = np.loadtxt(os.path.join(weight_path, 'proposed_weights(' + str(testmark) + ').txt'))
        true_price = price_file[:, 0]
        true_weight = weight_file[:, 0]
        pred_price = price_file[:, 1]
        pred_weight = weight_file[:, 1]
#        true_price = np.loadtxt('./datasets/knapsack/test_prices/test_prices(' + str(testmark) + ').txt')
#        true_weight = np.loadtxt('./datasets/knapsack/test_weights/test_weights(' + str(testmark) + ').txt')

        real_obj, real_selected_num = actual_obj(true_price, capacity, true_weight, n_instance=test_case_num)
    #    print(np.mean(real_obj))
        corr_obj_list = []
        train_loss_list = []

#        pred_sol_temp = np.loadtxt('./datasets/knapsack/results/0.21_700/CombOptNet_cap' + str(capacity) + '_sols(' + str(testmark) + ').txt')
        
        for testNum in range(test_case_num):
           cur_true_weight = np.zeros(item_num)
           cur_true_val = np.zeros(item_num)
           cur_pred_weight = np.zeros(item_num)
           cur_pred_val = np.zeros(item_num)
           for i in range(item_num):
               cur_true_weight[i] = true_weight[i+testNum*item_num]
               cur_true_val[i] = true_price[i+testNum*item_num]
#               cur_pred_weight[i] = pred_weight[i+testNum*item_num]
#               cur_pred_val[i] = pred_price[i+testNum*item_num]
               cur_pred_weight[i] = ReLUVal
               cur_pred_val[i] = ReLUVal
           
           train_loss = compute_training_loss(cur_true_val, cur_true_weight, cur_pred_val, cur_pred_weight, capacity)
           train_loss_list.append(train_loss)
           
           corrrlst = compute_single_correction_obj(cur_true_val, cur_true_weight, cur_pred_val, cur_pred_weight, capacity)
           corr_obj_list.append(corrrlst)

        print("avgTrainLoss: ", sum(np.array(train_loss_list))/test_case_num, "avgPReg: ", sum(abs(real_obj - np.array(corr_obj_list)))/test_case_num, "avgTOV: ", sum(real_obj)/test_case_num, "avgEOV: ", sum(corr_obj_list)/test_case_num)

    print("\n")
