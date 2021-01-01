def loglik(X, y, w):
    import numpy as np
    return np.sum(-y*(X@w) + np.log(1+np.exp(X@w)))


def reg_log(X, y, ite_max=100, lbd=1e-12, pos_contraint=False):
    """
    y \in 1,0 
    """
    import numpy as np

    def proj_on_pos(w):
        return np.array([x if x > 0 else 0 for x in w])

    tol = 1e-4
    N, d = X.shape
    y = np.array(y)

    w = np.zeros(d)  # see 4.4 of ESLII
    weights = [w]

    J = [loglik(X, y, w)]
    # print(f"J[0] = {J[0]}")
    old_J = J[0] + 1
    conv = False
    i = 0
    while(not conv):
        i = i + 1

        Xw = X @ w

        p = np.exp(Xw)/(1+np.exp(Xw))
        W = np.diag(p)
        regul = lbd*np.identity(d)
        descent = np.linalg.solve(X.T @ W @ X + regul, X.T@(y-p))
        # print(f"descent: {descent}")
        step = 1
        update = 0.1
        cur_w = w+step*descent

        if pos_contraint:
            cur_w = proj_on_pos(cur_w)

        # print(f"cur_w : {cur_w}")
        # print(f"J : {loglik(X,y,cur_w)}")

        while (loglik(X, y, cur_w) > J[-1]):
            step = step*update
            cur_w = w + step*descent
            if pos_contraint:
                cur_w = proj_on_pos(cur_w)
        # print(f"step : {step}")

        w = cur_w

        J.append(loglik(X, y, w))
        weights.append(w)

        if (i > ite_max):
            conv = True
        if ((old_J - J[-1]) < tol):
            conv = True
        else:
            old_J = J[-1]

    return w, J, weights
