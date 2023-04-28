import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import random
import argparse
import logging
import sys
import os

def get_logger(log_folder_path, tag):   
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    s_handler = logging.StreamHandler(sys.stdout)
    f_handler = logging.FileHandler(log_folder_path+"/"+tag+".log", mode='a+')

    s_format = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    f_format = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    s_handler.setFormatter(s_format)
    f_handler.setFormatter(f_format)

    logger.addHandler(s_handler)
    logger.addHandler(f_handler)
    return logger

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--L', type=float, required=True)
    parser.add_argument('--delta', type=float, required=True)
    parser.add_argument('--ratio', type=float, required=True)
    parser.add_argument('--num_iter', type=int, required=True)
    parser.add_argument('--num_trial', type=int, required=True)
    parser.add_argument('--alg_name', type=str, default=None)
    parser.add_argument('--delta_range', type=str, required=False, default=None)
    parser.add_argument('--L_range', type=str, required=False, default=None)
    
    hp, unknown = parser.parse_known_args()
    return hp

def f(x, L):
    return (L*x[0]**2+x[1]**2)/2

def g(x, L, delta):
    if delta ==0:
        return np.array([L*x[0], x[1]])        
    if delta>0:
        rand1 = np.random.normal(1, np.sqrt(delta/2), 1)[0]
        rand2 = np.random.normal(1, np.sqrt(delta/2), 1)[0]
    else:
        rand1 = 1
        rand2 = 1
    return np.array([rand1*L*x[0], rand2*x[1]])

def inti():
    x1 = random.gauss(1, 3)
    x2 = random.gauss(1, 3)
    return np.array([x1, x2])

def sgd(L, delta, num_iter, num_trial, ratio):
    loss = np.zeros(num_trial)
    x = np.zeros((num_trial, 2))
    
    step_size = 1/(L*(1+delta))
    
    for i in range(num_trial):
        x[i] = inti()
        loss[i] = f(x[i], L)
    inti_loss = loss.mean()
    
    for j in range(num_iter):
        for i in range(num_trial):
            if f(x[i], L)<1e8:
                grad = g(x[i], L, delta)
                x[i] -= step_size*grad
            loss[i] = f(x[i], L)
        loss_mean = loss.mean()
        if loss_mean<ratio*inti_loss:
            return j
        elif j>int(num_iter/4) and loss_mean>inti_loss:
            return np.inf
        elif loss_mean>1e8:
            return np.inf
    return num_iter

def nam(L, delta, num_iter, num_trial, ratio):
    loss = np.zeros(num_trial)
    y0 = np.zeros((num_trial, 2))
    y1 = np.zeros((num_trial, 2))
    
    step_size = (1-delta)/(L*(1+delta))
    beta = (1-np.sqrt(step_size))/(1+np.sqrt(step_size))
    
    for i in range(num_trial):
        y1[i] = inti()
        loss[i] = f(y1[i], L)
    inti_loss = loss.mean()
    
    for j in range(num_iter):
        for i in range(num_trial):
            if f(y1[i], L)<1e8:
                x = y1[i] + beta*(y1[i] - y0[i])
                grad = g(x, L, delta)
                temp = y1[i].copy()
                y1[i] = y1[i] + beta*(y1[i] - y0[i]) - step_size*grad
                y0[i] = temp
            loss[i] = f(y1[i], L)
        loss_mean = loss.mean()
        if loss_mean<ratio*inti_loss:
            return j
        elif j>int(num_iter/4) and loss_mean>inti_loss:
            return np.inf
        elif loss_mean>1e8:
            return np.inf
    return num_iter

def rmm(L, delta, num_iter, num_trial, ratio):
    if delta>0.25:
        return np.inf
    loss = np.zeros(num_trial)
    
    K = L
    if delta == 0:
        theta = 1
        rho = 1-1/np.sqrt(K)
    if delta>0:
        theta = 0.5-np.sqrt(delta)
        rho = 1-np.sqrt(2)*theta/np.sqrt(K)
    KK = L/theta
    beta = (KK*rho**3)/(KK - 1)
    eta = KK*(1-rho)*(1-rho**2)
    
    y = np.zeros((num_trial, 2))
    z = np.zeros((num_trial, 2))
    for i in range(num_trial):
        y[i] = inti()
        loss[i] = f(y[i], L)
    inti_loss = loss.mean()
    
    for j in range(num_iter):
        for i in range(num_trial):
            if f(y[i], L)<1e8:
                x = y[i] + beta/eta*z[i]
                grad = g(x, L, delta)
                z[i] = beta*z[i] - eta*grad/L
                y[i] = y[i] + z[i]
            loss[i] = f(y[i], L)
        loss_mean = loss.mean()
        if loss_mean<ratio*inti_loss:
            return j
        elif j>int(num_iter/2) and loss_mean>inti_loss:
            return np.inf
        elif loss_mean>1e8:
            return np.inf
    return num_iter

def dam(L, delta, num_iter, num_trial, ratio):
    loss = np.zeros(num_trial)
    y = np.zeros((num_trial, 2))
    z = np.zeros((num_trial, 2))
    
    eta = 1/((1+delta)*L)
    theta = 1/((1+delta)*np.sqrt(L))
    
    for i in range(num_trial):
        y[i] = inti()
        z[i] = y[i].copy()
        loss[i] = f(y[i], L)
    inti_loss = loss.mean()

    for j in range(num_iter):
        for i in range(num_trial):
            if f(y[i], L)<1e8:
                x = y[i]/(1+theta) + theta*z[i]/(1+theta)
                grad = g(x, L, delta)
                z[i] = theta*(x-grad) + (1-theta)*z[i]
                y[i] = x - eta*grad
            loss[i] = f(y[i], L)
        loss_mean = loss.mean()
        if loss_mean<ratio*inti_loss:
            return j
        elif j>int(num_iter/4) and loss_mean>inti_loss:
            return np.inf
        elif loss_mean>1e8:
            return np.inf
    return num_iter
    
def idam(L, delta, num_iter, num_trial, ratio):
    loss = np.zeros(num_trial)
    y = np.zeros((num_trial, 2))
    z = np.zeros((num_trial, 2))
    
    theta = np.sqrt((1-delta)/((1+delta)*L))
    
    for i in range(num_trial):
        y[i] = inti()
        z[i] = y[i].copy()
        loss[i] = f(y[i], L)
    inti_loss = loss.mean()

    for j in range(num_iter):
        for i in range(num_trial):
            if f(y[i], L)<1e8:
                x = y[i]/(1+theta) + theta*z[i]/(1+theta)
                grad = g(x, L, delta)
                z[i] = theta*x + (1-theta)*z[i]-theta*grad
                y[i] = (1-theta)*y[i]+theta*z[i]
            loss[i] = f(y[i], L)
        loss_mean = loss.mean()
        if loss_mean<ratio*inti_loss:
            return j
        elif j>int(num_iter/4) and loss_mean>inti_loss:
            return np.inf
        elif loss_mean>1e8:
            return np.inf
    return num_iter


def mc(alg_name, L, delta, num_iter, num_trial, ratio):    
    if alg_name == "sgd":
        alg = sgd
    elif alg_name == "nam":
        alg = nam
    elif alg_name == "rmm":
        alg = rmm
    elif alg_name == "dam":
        alg = dam
    elif alg_name == "idam":
        alg = idam
    else:
        raise Exception("Sorry, cannot find alg") 
    
    
    step_num = np.inf
    tmp = alg(L, delta, num_iter, num_trial, ratio)
    if tmp<step_num:
        step_num = tmp
    return step_num

def get_range(ran):
    res = [float(num) for num in ran.split(",")]
    return res
    
def run(hp):
    logger.info("create linspace of parameters")
    if hp.delta_range:
        delta_range = get_range(hp.delta_range)
        deltas = np.linspace(delta_range[0], delta_range[1], int(delta_range[2]), endpoint=False)
        logger.info("start MC")
        output = np.zeros(len(deltas))
        for idx, delta in enumerate(deltas):
            output[idx] = mc(hp.alg_name, hp.L, delta, hp.num_iter, hp.num_trial, hp.ratio)
            logger.info(f"complete {idx+1}/{len(deltas)} iterations, delta={delta:.8f}, alg_name={hp.alg_name}, output={output[idx]}")
        logger.info("save results")
        df = pd.DataFrame({"delta":deltas, "num_iters": output})
        df.to_csv(hp.log_dir+f"{hp.alg_name}.csv", index=False)
    else:
        L_range = get_range(hp.L_range)
        # Ls = np.linspace(L_range[0], L_range[1], int(L_range[2]), endpoint=False)
        Ls = np.linspace(L_range[0], L_range[1], int(L_range[2]), endpoint=False)
        # Ls = np.logspace(0.5, 2.5)
        logger.info("start MC")
        output = np.zeros(len(Ls))
        for idx, L in enumerate(Ls):
            output[idx] = mc(hp.alg_name, L, hp.delta, int(L*20), hp.num_trial, hp.ratio)
            logger.info(f"complete {idx+1}/{len(Ls)} iterations, L={L:.8f}, alg_name={hp.alg_name}, output={output[idx]}")        
        logger.info("save results")
        df = pd.DataFrame({"L":Ls, "num_iters": output})
        df.to_csv(hp.log_dir+f"{hp.alg_name}.csv", index=False)
    
    # plt.plot(deltas, output)
    # plt.xlabel("delta")
    # plt.ylabel("num of iterations")
    # plt.grid()
    # plt.title(f"alg: {hp.alg_name}, L:{hp.L}, num_iter:{hp.num_iter}, num_trial:{hp.num_trial}")
    # plt.savefig(hp.log_dir+f"{hp.alg_name}.png")
    
if __name__ == "__main__":
    hp = get_arguments()
    if not hp.log_dir:
        hp.log_dir = hp.alg_name
    hp.tag = str(time.time()).split(".")[0]
    create_folder(hp.log_dir)
    if hp.delta_range:
        hp.log_dir = hp.log_dir+"/L_"+str(int(hp.L))+"/"+hp.tag+"/"
    else:
        hp.log_dir = hp.log_dir+"/delta_"+str(hp.delta)+"/"+hp.tag+"/"
    create_folder(hp.log_dir)
    
    logger = get_logger(hp.log_dir, hp.tag)
    for k, v in vars(hp).items():
        logger.info(k+": "+str(v))
        
    run(hp)