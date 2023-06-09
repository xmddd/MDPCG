import numpy as np

def init_flow(env, g, m):
    """
    Init an flow allocation x for graph g.
    """
    N = env.N
    T = env.T
    g = env.g
    # pi = np.zeros((N, N, T))
    x = np.zeros((T, N, N))
    y = np.zeros((T, N))
    y[0] = m
    for t in range(T):
        for j in range(N):
            # outd = g.vs[j].outdegree()
            pi = np.zeros(N)
            list = [i.index for i in g.vs[j].successors()]
            for i in range(N):
                if i in list:
                    pi[i] = np.random.random(1)
                else:
                    pi[i] = 0
            pi = pi / np.sum(pi)
            x[t, j] = pi * y[t, j]
        if t < T-1:
            for j in range(N):
                y[t+1, j] = np.sum(x[t, :, j])

    return x

def ValueIteration(env, d, debug=False):
    """
    Calculate best response by cost d.
    """
    N = env.N
    T = env.T
    g = env.g

    MAX = 1000
    l = np.ones((T, N, N)) * MAX
    v = np.zeros((T, N))
    pi = np.zeros((T, N, N))
    for t in reversed(range(T)):
        for j in range(N):
            list = [i.index for i in g.vs[j].successors()]
            for i in list:
                eid = g.get_eid(j, i)
                if t == T-1:
                    l[t, j, i] = d[t, j, i]
                else:
                    l[t, j, i] = d[t, j, i] + v[t+1, i]
            v[t, j] = np.min(l[t, j, :])
            pi[t, j, np.argmin(l[t, j, :])] = 1
    return pi


def RetrieveDensity(env, pi, m):
    """
    Retrieve a flow from policy pi.
    """
    N = env.N
    T = env.T
    g = env.g

    x = np.zeros((T, N, N))
    for t in range(T):
        for i in range(N):
            # k = pi[t, i]
            list = [k.index for k in g.vs[i].successors()]
            for k in list:
                if t == 0:
                    x[t, i, k] = m[i] * pi[t, i, k]
                else:
                    x[t, i, k] = np.sum(x[t-1, :, i]) * pi[t, i, k]
    return x

def Regression(env, x, V, U, D):
    """
    Estimate congestion parameters by regularized least-squares method.
    """
    N = env.N
    T = env.T
    # g = env.g
    p = env.p

    
    theta = np.zeros((T,N,N,p+1,1))
    X = np.zeros((T,N,N,p+1,1))
    # X[:,:,:,:,0] = np.array([x ** k for k in range(p+1)]).T
    
    for t in range(T):
        for j in range(N):
            for i in range(N):
                X[t,j,i] = np.array([[x[t,j,i] ** k for k in range(p+1)]]).T
                V[t,j,i] = V[t,j,i] + X[t,j,i] @ X[t,j,i].T
                U[t,j,i] = U[t,j,i] + D[t,j,i] * X[t,j,i]
                # print(t,j,i)
                # print(np.linalg.inv(V[t,j,i]))
                # print(U[t,j,i])
                theta[t,j,i] = np.linalg.inv(V[t,j,i]) @ U[t,j,i]
                # print(theta[t,j,i])
                # d_estimate[t,j,i] = (theta[t,j,i].T @ X[t,j,i]).item()
    
    return theta, V, U

def Estimate(env, theta, x):
    N = env.N
    T = env.T
    # g = env.g
    p = env.p
    X = np.zeros((T,N,N,p+1,1))
    d_estimate = np.zeros((T,N,N))
    for t in range(T):
        for j in range(N):
            for i in range(N):
                X[t,j,i] = np.array([[x[t,j,i] ** k for k in range(p+1)]]).T
                d_estimate[t,j,i] = (theta[t,j,i].T @ X[t,j,i]).item()
    
    return d_estimate