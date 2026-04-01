# Adapted from HDMM (https://github.com/dpcomp-org/hdmm)
# Original author: Ryan McKenna
# Licensed under the GNU Affero General Public License v3.0 (AGPL-3.0)

from functools import reduce
import numpy as np
from scipy import optimize
from scipy import sparse
from scipy.sparse.linalg import spsolve_triangular
from scipy.linalg.lapack import dpotrf, dpotri


class McKennaConvex:
    def __init__(self, n):
        self.n = n
        self._mask = np.tri(n, dtype=bool, k=-1)
        self._params = np.zeros(n * (n - 1) // 2)
        self.X = np.zeros((n, n))

    def strategy(self):
        tri = np.zeros((self.n, self.n))
        tri[self._mask] = self._params
        X = np.eye(self.n) + tri + tri.T
        A = np.linalg.cholesky(X).T
        return A

    def _set_workload(self, W):
        # self.V = W.gram().dense_matrix().astype(float)
        self.V = W.T @ W
        self.W = W



    def _loss_and_grad(self, params):
        V = self.V
        X = self.X
        X.fill(0)
        # X = np.zeros((self.n,self.n))
        X[self._mask] = params
        X += X.T
        np.fill_diagonal(X, 1)

        zz, info0 = dpotrf(X, False, False)
        iX, info1 = dpotri(zz)
        iX = np.triu(iX) + np.triu(iX, k=1).T
        if info0 != 0 or info1 != 0:
            # print('checkpt')
            return self._loss * 100, np.zeros_like(params)

        loss = np.sum(iX * V)
        G = -iX @ V @ iX
        g = G[self._mask] + G.T[self._mask]

        self._loss = loss
        # print(np.sqrt(loss / self.W.shape[0]))
        return loss, g  # G.flatten()

    def optimize(self, W):
        self._set_workload(W)

        eig, P = np.linalg.eigh(self.V)
        eig = np.real(eig)
        eig[eig < 1e-10] = 0.0
        X = P @ np.diag(np.sqrt(eig)) @ P.T
        X /= np.diag(X).max()
        x = X[self._mask]

        # x = np.eye(self.n).flatten()
        # bnds = [(1,1) if x[i] == 1 else (None, None) for i in range(x.size)]
        # x = self._params

        opts = {'maxcor': 1}
        res = optimize.minimize(self._loss_and_grad, x, jac=True, method='L-BFGS-B', options=opts)
        self._params = res.x
        # print(res)
        return res.fun


if __name__ == "__main__":
    W = np.array([[1, 1, 1, 0,0,0], [0,0,0,1, 1, 1],
                  [1, 0, 0, 1,0,0], [0, 1, 0,0,1, 0],[0,0,1,0,0,1]])
    temp = McKennaConvex(6)
    fun = temp.optimize(W)
    strategy = temp.strategy()
    print(strategy)
