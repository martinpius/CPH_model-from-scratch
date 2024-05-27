import numpy as np 

def loglikelihood(T,E, X, betas):

    risk = np.dot(X, betas)
    ll = 0 

    for i in range(len(T)):
        if E[i] == 1:
            risk_set = risk[T>T[i]]
            max = np.max(risk_set)
            ll+=risk[i] - (max + np.log(np.sum(np.exp(risk_set - max))))
    return ll 

if __name__ == "__main__":
    # Testing the likelihood loss
    n, p = 10, 3
    true_betas = np.array([0.5, -0.2, 0.3])
    theta = 0.1 
    X = np.random.randn(n, p)
    scale = 1/(theta * np.exp(np.dot(X, true_betas)))
    T = np.random.exponential(scale = scale, size = n)
    C = np.random.exponential(scale = 2, size = n)
    E = T<C
    T = np.minimum(T, C)
    print(T.shape, E.shape, C.shape)
    ll = loglikelihood(T = T, E = E, X = X, betas = true_betas)
    print(ll)


