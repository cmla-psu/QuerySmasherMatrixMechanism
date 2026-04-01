import numpy as np
import itertools
import scipy.sparse as sp
from collections import defaultdict
import cvxpy as cp
from .softmax import *
from .hdmm_convex import McKennaConvex
import gurobipy as gp
from gurobipy import GRB
from .parameter import options


def find_var_sum_cauchy(v, p, c=1.0):
    """Solve the optimization problem in appendix B.6.

    min sum(v * sigma^2)
    s.t. sum(p / sigma^2) == c
    """
    T = np.sum(np.sqrt(v * p))**2 / c
    x = np.sqrt(T * p / (c * v))
    return x, T

def find_var_sum_cvxpy(var, pcost):
    """Solve the small optimization problem.

    min sum(1/x)
    s.t. A x <= b
    """
    size = len(var)
    x = cp.Variable(size)
    constraints = [x >= 0]
    constraints += [pcost @ cp.inv_pos(x) <= 1]
    obj = cp.Minimize(cp.sum(var @ x))
    prob = cp.Problem(obj,
                      constraints)
    prob.solve()
    print("obj sum of var:", obj.value)
    return x.value, obj.value


def find_var_sum_gurobi(var, pcost):
    """Solve the small optimization problem.

    min var @ x
    s.t. pcost @ (1/x) <= 1
    """
    env = gp.Env(empty=True)
    # env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model(env=env)
    m.Params.TIME_LIMIT = 10
    m.setParam('NonConvex', 2)
    # m.setParam(GRB.Param.OutputFlag, 0)
    size = len(var)
    x = m.addMVar(size, lb=1e-5)
    x_inv = m.addMVar(size, lb=1e-5)

    m.setObjective(var @ x, sense=GRB.MINIMIZE)
    m.addConstr(pcost @ x_inv <= 1)
    for i in range(size):
        m.addConstr(x[i] * x_inv[i] == 1.0)
    m.optimize()
    return x.X, var @ x.X


def find_var_max_cvxpy(coeff, A, b):
    """Solve the optimization problem.

    min sum(coeff / sigma^2)
    s.t. A sigma^2 <= b
    """
    size = len(coeff)
    x = cp.Variable(size)
    constraints = [x >= 0]
    constraints += [A @ x - b <= 0]
    obj = cp.Minimize(cp.sum(coeff @ cp.inv_pos(x)))
    prob = cp.Problem(obj, constraints)
    prob.solve()
    # print("obj:", obj.value)

    # scale the variable so that the privacy cost is 1
    return x.value * obj.value, obj.value

#
def find_var_max_gurobi(coeff, A, b, time_limit=10):
    """Solve the small optimization problem.

    min sum(1/x)
    s.t. A x <= b
    """
    env = gp.Env(params=options)
    env.setParam("OutputFlag", 0)
    env.start()
    m = gp.Model(env=env)
    #set time limit to inf
    #m.Params.TIME_LIMIT = time_limit
    m.setParam('NonConvex', 2)
    #m.setParam('TimeLimit', 5*60)
    # m.setParam(GRB.Param.OutputFlag, 0)
    size = np.shape(A)[1]
    x = m.addMVar(size, lb=1e-5)
    x_inv = m.addMVar(size, lb=1e-5)

    m.setObjective(coeff @ x_inv, sense=GRB.MINIMIZE)
    m.addConstr(A @ x <= b)
    for i in range(size):
        m.addConstr(x[i] * x_inv[i] == 1.0)
    m.optimize()
    obj = m.getObjective().getValue()
    return x.X * obj, obj


def subtract_matrix(k):
    """Return matrix C_k."""
    mat = np.zeros([k-1, k])
    for i in range(k-1):
        mat[i, i] = 1
        mat[i, i+1] = -1
    return mat


def subtract_matrix_v2(m):
    """
    Return an (m-1) x m matrix where the first column is all 1s, entries (i, i+1) are -1, and all other entries are 0.
    """
    mat = np.zeros((m-1, m))
    mat[:, 0] = 1
    for i in range(m-1):
        mat[i, i+1] = -1
    return mat


def all_subsets(att):
    """Return all subsets of a tuple."""
    length = len(att)
    subsets = [()]
    for i in range(1, length + 1):
        subset_i = list(itertools.combinations(att, i))
        subsets = subsets + subset_i
    return subsets


def sqrt_mat(X):
    """Calculate the decomposition X = B^T B."""
    vec, mat = np.linalg.eigh(X)
    idx = np.where(vec > 1e-5)[0]
    B = (mat[:, idx] * np.sqrt(vec[idx])).T
    return B


def haar_tree_residual(n):
    """
    Build the (n-1) x n residual part of the Haar binary tree.

    Row 1 (Bs = all-ones) is NOT included; only the "left − right" rows.

    For n=8 this produces:
      [ 1  1  1  1 -1 -1 -1 -1]   level 1: left half vs right half
      [ 1  1 -1 -1  0  0  0  0]   level 2
      [ 0  0  0  0  1  1 -1 -1]
      [ 1 -1  0  0  0  0  0  0]   level 3 (leaf pairs)
      [ 0  0  1 -1  0  0  0  0]
      [ 0  0  0  0  1 -1  0  0]
      [ 0  0  0  0  0  0  1 -1]

    Requires n to be a power of 2 and n >= 2.
    """
    assert n >= 2 and (n & (n - 1)) == 0, f"n must be a power of 2, got {n}"
    rows = []
    block_size = n // 2
    while block_size >= 1:
        num_blocks = n // (2 * block_size)
        for b in range(num_blocks):
            row = np.zeros(n)
            start = b * 2 * block_size
            row[start:start + block_size] = 1
            row[start + block_size:start + 2 * block_size] = -1
            rows.append(row)
        block_size //= 2
    return np.array(rows)


def prefix_workload(k):
    #TODO: find the meaning of it 
    mat = np.zeros([k, k])
    for i in range(k):
        mat[i, :i+1] = 1
    return mat

def range_workload(k):
    """
    Returns a matrix of shape (k*(k+1)//2, k),
    where each row corresponds to a contiguous
    range [i, j] (0-based), and columns i..j
    are set to 1 in that row.
    """
    num_ranges = k * (k + 1) // 2   # number of contiguous subranges
    mat = np.zeros((num_ranges, k))
    
    row = 0
    for start in range(k):
        for end in range(start, k):
            mat[row, start:end+1] = 1
            row += 1
    return mat


def _equalize_sub(Sub):
    """Reweight rows of Sub to equalize column L2 norms.

    Finds weights w such that all columns of diag(w) @ Sub have equal L2 norm.
    Solves Sub²ᵀ h ≈ 1 where h_l = w_l², then w = sqrt(h).

    If the unconstrained solution has non-positive entries, falls back to
    bounded least squares with a relative floor to prevent any row from
    being effectively removed (floor = max(h) * 0.01, giving at most
    a 10:1 ratio in row weights w).

    Returns the reweighted Sub matrix (Gamma remains I).
    """
    r, n = Sub.shape
    Sub2 = Sub ** 2
    A = Sub2.T  # n x r

    # Least-squares: A @ h ≈ 1
    h, _, _, _ = np.linalg.lstsq(A, np.ones(n), rcond=None)

    if np.any(h <= 0):
        # Relative floor: keep all rows with weight >= 10% of max
        # (h_floor / h_max = 0.01  ⟹  w_min / w_max = sqrt(0.01) = 0.1)
        h_floor = np.max(h) * 0.01 if np.max(h) > 0 else 1.0 / r
        from scipy.optimize import lsq_linear
        result = lsq_linear(A, np.ones(n), bounds=(h_floor, np.inf))
        h = result.x

    w = np.sqrt(h)
    return Sub * w[:, None]  # row-wise scaling


def find_residual_basis_sum(k, base):
    if base == 'P':
        work = prefix_workload(k)
    elif base == 'I':
        work = np.eye(k)
    elif base == 'R':
        work = range_workload(k)
    elif base == 'H':
        work = np.eye(k)
    else:
        work = None
    Bs = np.ones([1, k])

    # Algorithm 4, Step 1: Identity base
    if base == 'I':
        res_mat = subtract_matrix(k)
        gamma = res_mat.copy()  # Gamma_i = Sub_i for identity
        l_mat = np.concatenate([Bs, res_mat]).T
        r_mat = work.T
        X = np.linalg.solve(l_mat, r_mat).T
        Us = X[:, 0].reshape(-1, 1)
        Ur = X[:, 1:]
        return Bs, res_mat, Us, Ur, gamma

    # Algorithm 4, Step 2: Non-identity bases (P, R, H)
    if base == 'H':
        Si = np.concatenate([Bs, haar_tree_residual(k)])
    else:
        # Center workload and prepend sum query before McKennaConvex optimization
        proj = np.eye(k) - np.ones((k, 1)) @ np.ones((1, k)) / k
        projw = work @ proj
        workprime = np.vstack([Bs, projw])
        temp = McKennaConvex(k)
        temp.optimize(workprime)
        Si = temp.strategy()

    # Step 2a: P1 = C^{-1/2} (S - S 1 1^T / n), with C = I
    P1 = Si - Si @ Bs.T @ Bs / k

    # Step 2b: Cholesky of P1^T P1
    gram = P1.T @ P1
    gram += np.eye(k) * 1e-10  # regularize
    L = np.linalg.cholesky(gram)

    # Step 2c: linearly independent columns of L
    col_norms = np.linalg.norm(L, axis=0)
    keep_indices = np.argsort(col_norms)[-(k - 1):]
    keep_indices = np.sort(keep_indices)
    P2 = L[:, keep_indices]

    # Step 2d: Sub_i = P2^T
    res_mat = P2.T

    # Centered workload produces naturally balanced Sub; no reweighting needed.
    # Optional: res_mat = _equalize_sub(res_mat) for additional equalization.

    # Decompose workload into sum + residual components
    l_mat = np.concatenate([Bs, res_mat]).T
    r_mat = work.T
    X = np.linalg.solve(l_mat, r_mat).T
    Us = X[:, 0].reshape(-1, 1)
    Ur = X[:, 1:]

    gamma = np.eye(k - 1)  # Gamma = I since reweighting is absorbed into Sub

    return Bs, res_mat, Us, Ur, gamma

def find_residual_basis_max(k, base, method='cholesky'):
    """Build residual basis for MaxVar using matrix_query optimizer.

    method='cholesky': centered matrix_query + Algorithm 4 Cholesky (default)
    method='schur':    centered matrix_query + Schur complement
    method='old':      original uncentered matrix_query + Schur complement
    """
    if base == 'P':
        work = prefix_workload(k)
    elif base == 'I':
        work = np.eye(k)
    elif base == 'R':
        work = range_workload(k)
    else:
        work = None
    Bs = np.ones([1, k])

    # Center workload for cholesky/schur methods (not for identity base)
    if method in ('cholesky', 'schur') and base != 'I':
        proj = np.eye(k) - np.ones((k, 1)) @ np.ones((1, k)) / k
        projw = work @ proj
        opt_work = np.vstack([Bs, projw])
    else:
        opt_work = work

    param_m, param_n = opt_work.shape
    bound = np.ones(param_m)*1

    args = configuration()
    args.init_mat = 'id_index'
    args.maxitercg = 5
    args.theta = 1e-8
    args.sigma = 1e-8
    args.NNTOL = 1e-5
    args.TOL = 1e-5

    index = opt_work
    basis = np.eye(param_n)

    mat_opt = matrix_query(args, basis, index, bound)
    mat_opt.optimize()
    mat_cov = mat_opt.cov/np.max(mat_opt.f_var)

    pmat_CA = np.linalg.inv(mat_cov)

    if method == 'cholesky' and base != 'I':
        # Algorithm 4 Cholesky: use chol(precision) as strategy Si
        Si = np.linalg.cholesky(pmat_CA).T
        P1 = Si - Si @ Bs.T @ Bs / k
        gram = P1.T @ P1
        gram += np.eye(k) * 1e-10
        L = np.linalg.cholesky(gram)
        col_norms = np.linalg.norm(L, axis=0)
        keep_indices = np.argsort(col_norms)[-(k - 1):]
        keep_indices = np.sort(keep_indices)
        P2 = L[:, keep_indices]
        res_mat = P2.T
    else:
        # Schur complement (for identity base, or schur/old methods)
        A = np.linalg.cholesky(pmat_CA).T
        Ua = Bs @ np.linalg.pinv(A)
        var_sum_query = Ua @ Ua.T
        pmat_res = pmat_CA - 1.0 / var_sum_query
        res_mat = sqrt_mat(pmat_res)

    # Decompose original workload (not centered) into sum + residual
    l_mat = np.concatenate([Bs, res_mat]).T
    r_mat = work.T
    X = np.linalg.solve(l_mat, r_mat).T
    Us = X[:, 0].reshape(-1, 1)
    Ur = X[:, 1:]

    return Bs, res_mat, Us, Ur


def mult_kron_vec(mat_ls, vec):
    """Fast Kronecker matrix vector multiplication."""
    V = vec.reshape(-1, 1)
    row = 1
    X = V.T
    for Q in mat_ls[::-1]:
        m, n = Q.shape
        row *= m
        X = Q.dot(X.reshape(-1, n).T)
    return X.reshape(row, -1)

def calculate_range_workload(datavector):
    """
    Given a 1D datavector (array-like), compute the sum of datavector[i..j]
    for all 0 <= i <= j < n, and return those sums in a 1D NumPy array.

    Returns
    -------
    range_sums : ndarray of shape (n*(n+1)//2,)
        Each entry corresponds to the sum of a contiguous subrange of datavector.
        The subranges are enumerated in row-major order:
           [0,0], [0,1], [0,2], ..., [1,1], [1,2], ..., [2,2], etc.
    """
    # 1. Compute prefix sums
    prefix = np.cumsum(datavector)
    n = len(datavector)

    # 2. Allocate space for all contiguous-range sums
    #    Number of subranges = n*(n+1)//2
    range_sums = np.zeros(n*(n+1)//2, dtype=datavector.dtype)

    # 3. Fill in the range sums using prefix sums
    idx = 0
    for start in range(n):
        for end in range(start, n):
            if start == 0:
                # sum(datavector[0..end]) = prefix[end]
                range_sums[idx] = prefix[end]
            else:
                # sum(datavector[start..end]) = prefix[end] - prefix[start-1]
                range_sums[idx] = prefix[end] - prefix[start - 1]
            idx += 1

    return range_sums
    


if __name__ == "__main__":
    # coeff = np.array([1, 0.571])
    # A = np.array([[0.123, 0.877], [0.354, 0.645], [1.0, 0]])
    # b = np.array([1, 1, 1])

    coeff = np.array([1, 1/3.0])
    A = np.array([[0.25, 0.75], [1.0, 0]])
    b = np.array([1, 1])
    x, obj = find_var_max_gurobi(coeff, A, b)
    print("obj: ", obj)
