import numpy as np

def ValueIteration(env, D, debug=False):
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
                    l[t, j, i] = D[t, j, i]
                else:
                    l[t, j, i] = D[t, j, i] + v[t+1, i]
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

def Estimate(env, x, V, U, D):
    """
    Estimate expected cost by regularized least-squares method.
    """
    N = env.N
    T = env.T
    # g = env.g
    p = env.p

    d_estimate = np.zeros((T,N,N))
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
                d_estimate[t,j,i] = (theta[t,j,i].T @ X[t,j,i]).item()
    
    return d_estimate, V, U