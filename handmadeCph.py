import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd 
from typing import List, Tuple 

# Implementation of Cox proportional hazard model
# Creating a simple survival data

def madeup_dataset() -> Tuple[pd.DataFrame, List, np.ndarray]:
    
    np.random.seed(2910)
    # Assume 3 std normal covariates, sample size = 300
    n, p = 300, 3
    X = np.random.randn(n, p)

    # Creating/assuming the true parameters
    betas = np.array([1, -2, 1])

    # Assume a constant baseline hazard [exp]
    theta = 0.1


    # Assume the simplest distribution for the
    # survival time [exponential]
    #scale = (1 / (theta * np.exp(np.dot(X,betas))))
    T = np.random.exponential(scale=1/(theta * np.exp(np.dot(X, betas))), size=n)

    # Introducing censorship

    C = np.random.exponential(scale = 2, size = n) # shape == [n, 1]

    # Create an indicator/status variable
    #print(T.shape, C.shape)
    E = T < C # shape == [n, 1] with (0 or 1)
    #print(E.shape)
    # grab the survival time [whatever comes first]
    T = np.minimum(T, C) 

    # compile the dataset
    df = np.column_stack((T, E, X))

    df1 = pd.DataFrame(data = df, columns = ["T", "E", "X1", "X2", "X3"])
    df1["E"] = df1["E"].astype(int)
    return (df1, betas,df)

def plotDistributions(data: pd.DataFrame) -> None:

    plt.figure(figsize = (8, 8))
    fig, (ax1, ax2) = plt.subplots(1, 2)
    data[data["E"] == 0]["T"].plot.hist(ax = ax1, color = "fuchsia", label = "Censored")
    data[data["E"] == 1]["T"].plot.hist(ax = ax2, color = "gray", label = "Event")
    ax1.legend(loc = "best")
    ax2.legend(loc = "best")
    ax1.set_title("Censorship distribution")
    ax2.set_title("Event distribution")
    plt.show()
 

def loglikelihood(T,E, X, betas):

    risk = np.dot(X, betas)
    ll = 0 

    for i in range(len(T)):
        if E[i] == 1:
            risk_set = risk[T>T[i]]
            max = np.max(risk_set)
            ll+=risk[i] - (max + np.log(np.sum(np.exp(risk_set - max))))
    return ll 

def gradients(X, E, T, betas):

    risks = np.dot(X, betas)
    grads = np.zeros_like(betas)

    for i in range(len(T)):
        if E[i] == 1:
            risk_set = risks[T>=T[i]] # grab X_j.T Betas
            X_risk_set = X[T>=T[i]]
            a_max = np.max(risk_set)
            exp_risk_set = np.exp(risk_set - a_max)
            
            sum_exp_risk_set = np.sum(exp_risk_set)
            weighted_sum_X = np.dot(exp_risk_set, X_risk_set) / sum_exp_risk_set
            
            grads += X[i] - weighted_sum_X
            
    return grads

def hessianCPH(X, E, T, betas):

    risks = np.dot(X, betas)
    H = np.zeros((len(betas), len(betas)))

    for i in range(len(T)):
        if E[i] == 1:
            risk_set = risks[T>=T[i]]
            X_risk_set = X[T>=T[i]]
            a_max = np.max(risk_set)
            exp_risk_set = np.exp(risk_set - a_max)
            sum_exp_risk_set = np.sum(exp_risk_set)
            
            weight = exp_risk_set / sum_exp_risk_set
            
            H += np.outer(X[i], X[i]) - (np.dot(X_risk_set.T * weight,X_risk_set)) / sum_exp_risk_set
    return H

def fitting_cox(X, T, E, eps = 1e-5, max_iters = 4000, lr = 0.00000045):

    betas = np.random.uniform(low = 0, high = 1, size = X.shape[1]) # Initializing the parameters to 0
    for it in range(max_iters):
        grads = gradients(X = X, E = E, T = T, betas = betas)
        # H = hessianCPH(X = X, E = E, T = T, betas = betas)
        # delta = np.linalg.solve(H, grads)
        # Adjust the step size
        betas += -lr * grads
        
        ll = loglikelihood(T = T, E = E, X = X, betas = betas)
        print(f">>>> iteration:{it}:loss: {-ll:.4f} ")
        if np.linalg.norm(lr * grads) < eps:
            break
    
    return betas


if __name__ == "__main__":
    res = madeup_dataset()
    data, betas, df_n = res
    betas_ = np.array([0.2, -0.4, 0.8])
    T, E, X = df_n[:,0], df_n[:,1], df_n[:, 2:]

    # ll = loglikelihood(T = T, E = E, X = X, betas = betas_)
    # print(ll)
    # grads = gradients(X = X, E = E, T = T, betas = betas_)
    # print(grads)
    hessian = hessianCPH(X = X, E = E, T = T, betas = betas_)
    print(f"Hessian: \n {hessian}")

    beta_hat = fitting_cox(X = X, T = T, E = E)
    print(f"\n >>>> beta estimates: {beta_hat}")
  