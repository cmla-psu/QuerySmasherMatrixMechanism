"""
Shared utilities for QuerySmasher experiments.

Provides:
  - RP+ integration (mock gurobipy, covariance computation, variance evaluation)
  - QS variance helper (run Smasher pipeline, return sum-of-variance)
  - Standard workload builders (random, identity, circular range, prefix, range, 2D range)
  - Data generation
"""

import sys
import os
import types
import numpy as np
import itertools
from functools import reduce

# --- Add project root to path ---
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smasher import Smasher
from query import MatrixQuery, KronQuery


# ======================================================================
# RP+ integration
# ======================================================================

def _ensure_rpp():
    """Mock gurobipy and import RP+ (lazy, idempotent)."""
    if "gurobipy" not in sys.modules or not hasattr(sys.modules["gurobipy"], "Model"):
        _mock = types.ModuleType("gurobipy")
        _mock.GRB = types.SimpleNamespace()
        sys.modules["gurobipy"] = _mock
    rpp_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "baselines")
    if rpp_path not in sys.path:
        sys.path.insert(0, rpp_path)


def all_subsets(att):
    """Generate all subsets of att (including empty set)."""
    s = list(att)
    return list(itertools.chain.from_iterable(
        itertools.combinations(s, r) for r in range(len(s) + 1)))


def rpp_marginal_covariance(rpp, domains, att, bases=None):
    """
    Build the true covariance matrix Cov(x_hat) for a reconstructed marginal.

    Noise model depends on basis type per the corrected RP+ formulation:
      - 'I' (original subtract_matrix): Sigma = sigma^2 * CC^T  (paper formula)
      - 'P' (custom McKennaConvex strategy): Sigma = sigma^2 * I  (iid noise)
    """
    if bases is None:
        bases = getattr(rpp, 'bases', ["I"] * len(domains))
    k = int(np.prod([domains[i] for i in att]))
    Cov = np.zeros((k, k))
    for subset in all_subsets(att):
        nl = rpp.res_dict[subset].noise_level
        recon_list, noise_list = [], []
        for at in att:
            d = domains[at]
            if at in subset:
                recon_list.append(rpp.residual_pinv[d])
                if bases[at] == 'I':
                    # Original subtract_matrix: noise cov = CC^T
                    noise_list.append(rpp.residual_matrix[d])
                else:
                    # Custom strategy (P): iid noise, Sigma = I
                    n_rows = rpp.residual_matrix[d].shape[0]
                    noise_list.append(np.eye(n_rows))
            else:
                recon_list.append(np.ones((d, 1)) / d)
                noise_list.append(np.eye(1))
        R = reduce(np.kron, recon_list)
        N_mat = reduce(np.kron, noise_list)
        Cov += nl * R @ N_mat @ N_mat.T @ R.T
    return Cov


def rpp_variance(domains, marginals, workload, bases=None):
    """
    Compute the sum-of-variance that RP+ achieves on a given workload.

    Uses corrected noise model per advisor guidance:
      - 'I' basis (original subtract_matrix): Sigma = sigma^2 * CC^T
      - 'P' basis (custom McKennaConvex strategy): Sigma = sigma^2 * I
    Noise allocation is re-solved with corrected pcost/var per attribute.

    Args:
        bases: per-attribute basis list (e.g. ["P","P","I"]).
               Defaults to ["I"] * len(domains).
    """
    _ensure_rpp()
    from resplan import ResPlanSum
    from resplan.utils import find_residual_basis_sum, find_var_sum_cauchy
    if bases is None:
        bases = ["I"] * len(domains)
    rpp = ResPlanSum(domains, bases=bases)
    for subset in marginals:
        rpp.input_mech(subset)

    # --- Correct pcost_res and var_res for 'P' basis attributes ---
    # For 'I' basis: paper formulas are correct (CC^T noise model)
    # For 'P' basis: use iid noise model → pcost = max(diag(R^T R)),
    #   var_res = trace(Ur Ur^T) instead of ||Ur R||^2
    for i, (k, base) in enumerate(zip(domains, bases)):
        if base != 'I':
            Bs, R, Us, Ur, _gamma = find_residual_basis_sum(k, base)
            rpp.pcost_res[i] = np.max(np.diag(R.T @ R))
            rpp.var_res[i] = np.trace(Ur @ Ur.T)
            rpp.var_sum[i] = np.trace(Us @ Us.T)

    # Recompute var_coeff_sum with corrected var_res/var_sum
    from collections import defaultdict
    rpp.var_coeff_sum = defaultdict(int)
    for att_key in rpp.marg_dict:
        att_subsets = all_subsets(att_key)
        for subset in att_subsets:
            var_res_list = [rpp.var_res[at] for at in subset]
            var_res_query = np.prod(var_res_list)
            var_sum_list = []
            for c in att_key:
                if c not in subset:
                    var_sum_list.append(rpp.var_sum[c])
            var_sum_query = np.prod(var_sum_list) if var_sum_list else 1.0
            rpp.var_coeff_sum[subset] += var_sum_query * var_res_query

    # Recompute pcost_coeff with corrected pcost_res
    for subset in rpp.res_dict:
        pcost_res_list = [rpp.pcost_res[at] for at in subset]
        rpp.pcost_coeff[subset] = np.prod(pcost_res_list)

    # Re-solve Cauchy-Schwarz with corrected v, p
    rpp.get_noise_level()

    # --- Fast path for uniform domains: group marginals by order ---
    # All k-way marginals on [n]^d have the same Cov and W, so compute once per group.
    from collections import Counter
    order_counts = Counter(len(s) for s in marginals)
    unique_domains = set(domains)
    all_same_W = all(
        np.array_equal(workload[marginals[0]], workload[s])
        for s in marginals if len(s) == len(marginals[0])
    ) if len(unique_domains) == 1 else False

    if len(unique_domains) == 1 and all_same_W:
        # Uniform domains: compute one representative per order
        total_var = 0.0
        representatives = {}
        for subset in marginals:
            k = len(subset)
            if k not in representatives:
                representatives[k] = subset
        for k, rep_subset in representatives.items():
            W = workload[rep_subset]
            Cov = rpp_marginal_covariance(rpp, domains, rep_subset, bases=bases)
            var_one = np.sum((W @ Cov) * W)
            total_var += var_one * order_counts[k]
        return total_var

    # --- Slow path: general case ---
    total_var = 0.0
    for subset in marginals:
        W = workload[subset]
        Cov = rpp_marginal_covariance(rpp, domains, subset, bases=bases)
        total_var += np.sum((W @ Cov) * W)
    return total_var


def rpp_variance_general(domains, marginals, workload, bases=None):
    """
    Compute RP+ sum-of-variance with workload-aware noise allocation.

    Unlike rpp_variance() which uses Kronecker product formulas for variance
    coefficients, this function computes var_coeff_sum by explicitly projecting
    the actual workload onto each residual subspace.  This lets RP+'s
    Cauchy-Schwarz noise allocation optimize for non-Kronecker workloads
    (affine, abs_diff) rather than assuming separable structure.

    Privacy cost (pcost_coeff) is domain-dependent and unchanged.
    Final variance evaluation uses rpp_marginal_covariance() (exact for any W).
    """
    _ensure_rpp()
    from resplan import ResPlanSum
    from resplan.utils import find_residual_basis_sum

    if bases is None:
        bases = ["I"] * len(domains)

    rpp = ResPlanSum(domains, bases=bases)
    for subset in marginals:
        rpp.input_mech(subset)

    # --- Correct pcost_res and var_res for non-'I' bases (same as rpp_variance) ---
    for i, (k, base) in enumerate(zip(domains, bases)):
        if base != 'I':
            Bs, R, Us, Ur, _gamma = find_residual_basis_sum(k, base)
            rpp.pcost_res[i] = np.max(np.diag(R.T @ R))
            rpp.var_res[i] = np.trace(Ur @ Ur.T)
            rpp.var_sum[i] = np.trace(Us @ Us.T)

    # Recompute pcost_coeff (domain-only, same as rpp_variance)
    for subset in rpp.res_dict:
        pcost_res_list = [rpp.pcost_res[at] for at in subset]
        rpp.pcost_coeff[subset] = np.prod(pcost_res_list)

    # --- Recompute var_coeff_sum using RECONSTRUCTION MATRICES ---
    # For each subspace A, compute ||W @ Recon_A @ N_A||_F^2 summed over marginals.
    # Recon_A = kron(recon_i) where recon_i = residual_pinv if i in A, else (1/d)*ones.
    # N_A = kron(noise_i) where noise_i = I for 'P' basis, residual_matrix for 'I' basis.
    # This correctly handles non-Kronecker workloads (affine, abs_diff).
    from collections import defaultdict, Counter
    rpp.var_coeff_sum = defaultdict(float)

    # Fast path for uniform domains: compute one representative per marginal order,
    # then multiply by count.  On [n]^d with the same W per order, all k-way marginals
    # contribute identically.  A specific subset S of order m receives contributions
    # from C(d-m, k-m) different k-way marginals (those containing S).
    unique_domains = set(domains)
    unique_bases = set(bases)
    # Check per-order workload equality
    order_reps = {}
    per_order_same = True
    if len(unique_domains) == 1 and len(unique_bases) == 1:
        for att_key in rpp.marg_dict:
            k = len(att_key)
            if k not in order_reps:
                order_reps[k] = att_key
            else:
                if not np.array_equal(workload[att_key], workload[order_reps[k]]):
                    per_order_same = False
                    break
    else:
        per_order_same = False

    if per_order_same and order_reps:
        from math import comb
        d_total = len(domains)
        _order_vals = {}  # order m -> var_coeff_sum value per m-subset
        # For each order-k representative, pick ONE m-subset per order m,
        # compute its contribution, multiply by C(d-m, k-m).
        for k, rep_key in order_reps.items():
            W = np.asarray(workload[rep_key])
            seen_orders = set()
            for subset in all_subsets(rep_key):
                m = len(subset)
                if m in seen_orders:
                    continue  # only one representative per (k, m) pair
                seen_orders.add(m)
                recon_factors = []
                noise_factors = []
                for at in rep_key:
                    d = domains[at]
                    if at in subset:
                        recon_factors.append(rpp.residual_pinv[d])
                        if bases[at] == 'I':
                            noise_factors.append(rpp.residual_matrix[d])
                        else:
                            n_rows = rpp.residual_matrix[d].shape[0]
                            noise_factors.append(np.eye(n_rows))
                    else:
                        recon_factors.append(np.ones((d, 1)) / d)
                        noise_factors.append(np.eye(1))

                Recon = recon_factors[0]
                for r in recon_factors[1:]:
                    Recon = np.kron(Recon, r)
                N_mat = noise_factors[0]
                for nn in noise_factors[1:]:
                    N_mat = np.kron(N_mat, nn)

                W_R_N = W @ Recon @ N_mat
                val_one = np.sum(W_R_N ** 2)
                # How many k-way marginals contain a given m-subset?
                count = comb(d_total - m, k - m)
                _order_vals[m] = _order_vals.get(m, 0.0) + val_one * count

        # Assign computed values to ALL subsets in res_dict by order
        for subset_key in rpp.res_dict:
            m = len(subset_key)
            if m in _order_vals:
                rpp.var_coeff_sum[subset_key] = _order_vals[m]
    else:
        # Slow path: iterate over all marginals
        for att_key in rpp.marg_dict:
            W = np.asarray(workload[att_key])
            att_subsets = all_subsets(att_key)

            for subset in att_subsets:
                recon_factors = []
                noise_factors = []
                for at in att_key:
                    d = domains[at]
                    if at in subset:
                        recon_factors.append(rpp.residual_pinv[d])
                        if bases[at] == 'I':
                            noise_factors.append(rpp.residual_matrix[d])
                        else:
                            n_rows = rpp.residual_matrix[d].shape[0]
                            noise_factors.append(np.eye(n_rows))
                    else:
                        recon_factors.append(np.ones((d, 1)) / d)
                        noise_factors.append(np.eye(1))

                Recon = recon_factors[0]
                for r in recon_factors[1:]:
                    Recon = np.kron(Recon, r)
                N_mat = noise_factors[0]
                for nn in noise_factors[1:]:
                    N_mat = np.kron(N_mat, nn)

                W_R_N = W @ Recon @ N_mat
                rpp.var_coeff_sum[subset] += np.sum(W_R_N ** 2)

    # Re-solve Cauchy-Schwarz with workload-aware coefficients
    rpp.get_noise_level()

    # Evaluate variance using exact covariance matrix
    # Fast path for uniform domains: group by order, compute once per group
    from collections import Counter
    order_counts = Counter(len(s) for s in marginals)
    unique_domains = set(domains)
    all_same_W = all(
        np.array_equal(workload[marginals[0]], workload[s])
        for s in marginals if len(s) == len(marginals[0])
    ) if len(unique_domains) == 1 else False

    if len(unique_domains) == 1 and all_same_W:
        total_var = 0.0
        representatives = {}
        for subset in marginals:
            k = len(subset)
            if k not in representatives:
                representatives[k] = subset
        for k, rep_subset in representatives.items():
            W = np.asarray(workload[rep_subset])
            Cov = rpp_marginal_covariance(rpp, domains, rep_subset, bases=bases)
            var_one = np.sum((W @ Cov) * W)
            total_var += var_one * order_counts[k]
        return total_var

    # Slow path: general case
    total_var = 0.0
    for subset in marginals:
        W = np.asarray(workload[subset])
        Cov = rpp_marginal_covariance(rpp, domains, subset, bases=bases)
        total_var += np.sum((W @ Cov) * W)
    return total_var


# ======================================================================
# QS variance helper
# ======================================================================

def qs_variance(domains, marginals, workload, solver_name="fourier",
                convex_optimizer="lbfgsb_fourier", convex_kwargs=None):
    """
    Run QS through the full Smasher pipeline and return sum-of-variance.

    Args:
        domains: list of domain sizes
        marginals: list of attribute subsets, e.g. [(0,), (1,), (0,1)]
        workload: dict mapping subset -> query matrix (n_queries x marginal_size)
        solver_name: "fourier" or "convex"
        convex_optimizer: optimizer for convexSolver ('lbfgsb', 'lbfgsb_fourier', 'scs')
        convex_kwargs: extra keyword args for convexSolver (mu_schedule, ftol, maxcor)
    """
    qs = Smasher(domains, privacy_cost=1.0, default_solver=solver_name,
                 noise=True, verbose=False, convex_optimizer=convex_optimizer,
                 convex_kwargs=convex_kwargs)
    for subset in marginals:
        qs.add_queries_to_workload([MatrixQuery(domains, subset, workload[subset])])
    qs.optimize()
    return qs.workload_variance()


# ======================================================================
# Data generation
# ======================================================================

def make_data_vector_and_df(domains, seed=42):
    """Create matching data for both QS (count vector) and RP+ (DataFrame)."""
    import pandas as pd
    np.random.seed(seed)
    total_cells = int(np.prod(domains))
    counts = np.random.randint(1, 50, size=total_cells).astype(float)
    records = []
    for flat_idx in range(total_cells):
        multi = np.unravel_index(flat_idx, tuple(domains))
        for _ in range(int(counts[flat_idx])):
            records.append(list(multi))
    col_names = [f"a{i}" for i in range(len(domains))]
    df = pd.DataFrame(records, columns=col_names)
    return counts, df, col_names


# ======================================================================
# Workload builders
# ======================================================================

def make_random_workload(domains, marginals, p, seed=0):
    """
    Random Bernoulli workload: for each marginal with k cells,
    generate 3k queries with entries independently 1 w.p. p.
    """
    rng = np.random.RandomState(seed)
    workload = {}
    for subset in marginals:
        k = int(np.prod([domains[i] for i in subset]))
        n_queries = 3 * k
        W = (rng.rand(n_queries, k) < p).astype(float)
        workload[subset] = W
    return workload


def make_identity_workload(domains, marginals):
    """Identity workload: I_k for each marginal with k cells."""
    workload = {}
    for subset in marginals:
        k = int(np.prod([domains[i] for i in subset]))
        workload[subset] = np.eye(k)
    return workload


def circular_range_queries(d):
    """
    All circular range queries on domain d.
    For each start s and length l (1..d): cells s, s+1, ..., s+l-1 (mod d).
    Returns d^2 x d binary matrix.
    """
    rows = []
    for s in range(d):
        for l in range(1, d + 1):
            row = np.zeros(d)
            for j in range(l):
                row[(s + j) % d] = 1.0
            rows.append(row)
    return np.array(rows)


def prefix_queries(d):
    """
    Prefix queries on domain d: query k sums cells 0..k-1 for k=1..d.
    Returns d x d lower-triangular binary matrix.
    """
    W = np.zeros((d, d))
    for k in range(d):
        W[k, :k + 1] = 1.0
    return W


def range_queries_1d(d):
    """
    All (non-circular) range queries on domain d.
    For each start s and end e (s <= e): cells s..e.
    Returns d*(d+1)/2 x d binary matrix.
    """
    rows = []
    for s in range(d):
        for e in range(s, d):
            row = np.zeros(d)
            row[s:e + 1] = 1.0
            rows.append(row)
    return np.array(rows)


def cross_product_prefix_2d(d1, d2):
    """
    Cross-product prefix queries on d1 x d2.
    Returns (d1*d2, d1*d2) matrix = kron(prefix(d1), prefix(d2)).
    """
    return np.kron(prefix_queries(d1), prefix_queries(d2))


def affine_queries_2d(d1, d2):
    """
    Affine threshold queries: {(i,j) : i+j <= c} for each c in [0, d1+d2-2].
    Returns (d1+d2-1, d1*d2) binary matrix.
    """
    rows = []
    for c in range(d1 + d2 - 1):
        row = np.zeros(d1 * d2)
        for i in range(d1):
            for j in range(d2):
                if i + j <= c:
                    row[i * d2 + j] = 1.0
        rows.append(row)
    return np.array(rows)


def abs_diff_queries_2d(d1, d2):
    """
    Absolute-difference threshold queries: {(i,j) : |i - j| <= c}
    for each c in [0, max(d1, d2) - 1].
    Returns (max(d1, d2), d1*d2) binary matrix.
    """
    max_d = max(d1, d2)
    rows = []
    for c in range(max_d):
        row = np.zeros(d1 * d2)
        for i in range(d1):
            for j in range(d2):
                if abs(i - j) <= c:
                    row[i * d2 + j] = 1.0
        rows.append(row)
    return np.array(rows)


def cross_product_ranges_2d(d1, d2):
    """
    2D cross-product range queries: all rectangles [s1..e1] x [s2..e2].
    Returns matrix of shape (n_queries, d1*d2).
    """
    ranges1 = []
    for s in range(d1):
        for e in range(s, d1):
            r = np.zeros(d1)
            r[s:e + 1] = 1.0
            ranges1.append(r)
    ranges2 = []
    for s in range(d2):
        for e in range(s, d2):
            r = np.zeros(d2)
            r[s:e + 1] = 1.0
            ranges2.append(r)
    rows = []
    for r1 in ranges1:
        for r2 in ranges2:
            rows.append(np.kron(r1, r2))
    return np.array(rows)


# ======================================================================
# Marginal helpers
# ======================================================================

def k_way_marginals(domains, max_k):
    """Generate all 1-way through max_k-way marginals."""
    att = tuple(range(len(domains)))
    marginals = []
    for k in range(1, max_k + 1):
        for subset in itertools.combinations(att, k):
            marginals.append(subset)
    return marginals


# ======================================================================
# Dataset definitions (real datasets from RP+ experiments)
# ======================================================================

DATASETS = {
    "Adult": {
        "domains": [100, 100, 100, 99, 85, 42, 16, 15, 9, 7, 6, 5, 2, 2],
        "num_attrs": 5,   # first 5 are numerical (P basis), rest categorical (I basis)
    },
    "CPS": {
        "domains": [50, 100, 7, 4, 2],
        "num_attrs": 2,   # first 2 are numerical
    },
    "Loans": {
        "domains": [101, 101, 101, 101, 3, 8, 36, 6, 51, 4, 5, 15],
        "num_attrs": 4,   # first 4 are numerical
    },
}


def identity_marginal_workload(domains, marginals, max_marg_size=5000):
    """Identity workload on each marginal, skip if marginal > max_marg_size."""
    workload = {}
    for subset in marginals:
        marg_size = int(np.prod([domains[i] for i in subset]))
        if 0 < marg_size <= max_marg_size:
            workload[subset] = np.eye(marg_size)
    return workload


# ======================================================================
# Composite workload builders (structured queries across marginals)
# ======================================================================

# Max 2D range queries before falling back to cross-product prefix
RANGE_QUERY_LIMIT = 50000


def build_prefix_workload(domains, marginals):
    """Prefix queries: 1-way prefix, 2-way cross-product prefix, 3+way identity."""
    workload = {}
    for subset in marginals:
        if len(subset) == 1:
            workload[subset] = prefix_queries(domains[subset[0]])
        elif len(subset) == 2:
            d1, d2 = domains[subset[0]], domains[subset[1]]
            workload[subset] = cross_product_prefix_2d(d1, d2)
        else:
            marg_size = int(np.prod([domains[i] for i in subset]))
            workload[subset] = np.eye(marg_size)
    return workload


def build_range_workload(domains, marginals):
    """Range queries: 1-way range, 2-way cross-product range (prefix fallback if >RANGE_QUERY_LIMIT), 3+way identity."""
    workload = {}
    for subset in marginals:
        if len(subset) == 1:
            workload[subset] = range_queries_1d(domains[subset[0]])
        elif len(subset) == 2:
            d1, d2 = domains[subset[0]], domains[subset[1]]
            n_range = (d1 * (d1 + 1) // 2) * (d2 * (d2 + 1) // 2)
            if n_range < RANGE_QUERY_LIMIT:
                workload[subset] = cross_product_ranges_2d(d1, d2)
            else:
                workload[subset] = cross_product_prefix_2d(d1, d2)
        else:
            marg_size = int(np.prod([domains[i] for i in subset]))
            workload[subset] = np.eye(marg_size)
    return workload


def cross_product_circ_ranges_2d(d1, d2):
    """
    2D cross-product circular range queries: kron(circ(d1), circ(d2)).
    Returns (d1^2 * d2^2, d1*d2) binary matrix.
    """
    return np.kron(circular_range_queries(d1), circular_range_queries(d2))


def build_circ_range_workload(domains, marginals):
    """Circular range queries: 1D circular, 2-way cross-product circular, 3+way identity."""
    workload = {}
    for subset in marginals:
        if len(subset) == 1:
            workload[subset] = circular_range_queries(domains[subset[0]])
        elif len(subset) == 2:
            d1, d2 = domains[subset[0]], domains[subset[1]]
            n_circ = d1**2 * d2**2
            if n_circ <= RANGE_QUERY_LIMIT:
                workload[subset] = cross_product_circ_ranges_2d(d1, d2)
            else:
                # Too many queries; fall back to cross-product prefix
                workload[subset] = cross_product_prefix_2d(d1, d2)
        else:
            marg_size = int(np.prod([domains[i] for i in subset]))
            workload[subset] = np.eye(marg_size)
    return workload


def build_affine_workload(domains, marginals):
    """Affine queries: 1-way prefix, 2-way affine, 3+way identity."""
    workload = {}
    for subset in marginals:
        if len(subset) == 1:
            workload[subset] = prefix_queries(domains[subset[0]])
        elif len(subset) == 2:
            d1, d2 = domains[subset[0]], domains[subset[1]]
            workload[subset] = affine_queries_2d(d1, d2)
        else:
            marg_size = int(np.prod([domains[i] for i in subset]))
            workload[subset] = np.eye(marg_size)
    return workload


def build_abs_diff_workload(domains, marginals):
    """Abs_diff queries: 1-way prefix, 2-way abs_diff, 3+way identity."""
    workload = {}
    for subset in marginals:
        if len(subset) == 1:
            workload[subset] = prefix_queries(domains[subset[0]])
        elif len(subset) == 2:
            d1, d2 = domains[subset[0]], domains[subset[1]]
            workload[subset] = abs_diff_queries_2d(d1, d2)
        else:
            marg_size = int(np.prod([domains[i] for i in subset]))
            workload[subset] = np.eye(marg_size)
    return workload


def build_random_workload(domains, marginals, p=0.3, seed=42):
    """Random Bernoulli(p) workload, 3k queries per marginal."""
    return make_random_workload(domains, marginals, p=p, seed=seed)


def build_identity_workload(domains, marginals):
    """Identity workload: I_k for each marginal."""
    return make_identity_workload(domains, marginals)


def build_mixed_basis_workload(domains, marginals, num_attrs):
    """Prefix basis for numerical attributes, identity for categorical.

    Matches RP+ paper setup: per-marginal workload is kron(B_i for i in subset)
    where B_i = prefix_queries(d_i) if i < num_attrs, else I(d_i).
    """
    workload = {}
    for subset in marginals:
        matrices = []
        for i in subset:
            d = domains[i]
            if i < num_attrs:
                matrices.append(prefix_queries(d))
            else:
                matrices.append(np.eye(d))
        W = reduce(np.kron, matrices)
        workload[subset] = W
    return workload


def max_2way_r(domains):
    """Maximum residual dimension r = prod(d_i - 1) across all 2-way partitions."""
    max_r = 0
    for i, j in itertools.combinations(range(len(domains)), 2):
        r = (domains[i] - 1) * (domains[j] - 1)
        max_r = max(max_r, r)
    return max_r


# ======================================================================
# Kronecker-structured workload builders
# ======================================================================

def build_kron_identity_workload(domains, marginals):
    """Identity workload as KronQuery: I(d_1) ⊗ I(d_2) ⊗ ... for each marginal."""
    workload = {}
    for subset in marginals:
        factors = [np.eye(domains[i]) for i in subset]
        workload[subset] = KronQuery(domains, subset, factors)
    return workload


def build_mixed_workload_v2(domains, marginals):
    """Mixed workload for synthetic [n]^d:
    1-way = range, 2-way = affine (dense), 3-way = prefix (Kron).

    Returns dict mapping subset -> query object.
    1-way and 2-way entries are numpy arrays (for MatrixQuery).
    3-way entries are KronQuery objects.
    """
    workload = {}
    for subset in marginals:
        if len(subset) == 1:
            workload[subset] = range_queries_1d(domains[subset[0]])
        elif len(subset) == 2:
            d1, d2 = domains[subset[0]], domains[subset[1]]
            workload[subset] = affine_queries_2d(d1, d2)
        else:
            # 3-way+: Kron prefix factors
            factors = [prefix_queries(domains[i]) for i in subset]
            workload[subset] = KronQuery(domains, subset, factors)
    return workload


def build_mixed_type_workload(domains, marginals, num_attrs):
    """Mixed-type workload for real datasets with affine queries on numeric pairs.

    Query type depends on attribute types (numeric vs categorical):
      - 1-way numeric: prefix_queries(d)
      - 1-way categorical: identity(d)
      - 2-way both numeric: affine_queries_2d(d1, d2) if r <= 5000,
        else kron(prefix, prefix)
      - 2-way other: kron(per-attr basis)  [KronQuery]
      - 3-way+: kron per-attr basis (prefix for numeric, identity for cat)

    Returns dict mapping subset -> query object.
    Dense queries returned as numpy arrays, Kron queries as KronQuery.
    """
    workload = {}
    for subset in marginals:
        num_indices = [i for i in subset if i < num_attrs]
        cat_indices = [i for i in subset if i >= num_attrs]
        n_num = len(num_indices)

        if len(subset) == 1:
            i = subset[0]
            d = domains[i]
            if i < num_attrs:
                workload[subset] = prefix_queries(d)
            else:
                workload[subset] = np.eye(d)

        elif len(subset) == 2:
            if n_num == 2:
                # Both numeric: affine if residual dim is tractable
                d1, d2 = domains[subset[0]], domains[subset[1]]
                r = (d1 - 1) * (d2 - 1)
                if r <= 5000:
                    workload[subset] = affine_queries_2d(d1, d2)
                else:
                    # Too large for dense SDP: fall back to kron(prefix, prefix)
                    factors = [prefix_queries(d1), prefix_queries(d2)]
                    workload[subset] = KronQuery(domains, subset, factors)
            else:
                # Mixed or both cat: kron per-attr basis
                factors = []
                for i in subset:
                    d = domains[i]
                    factors.append(prefix_queries(d) if i < num_attrs else np.eye(d))
                workload[subset] = KronQuery(domains, subset, factors)

        else:
            # 3-way or higher: always kron per-attr basis
            factors = []
            for i in subset:
                d = domains[i]
                factors.append(prefix_queries(d) if i < num_attrs else np.eye(d))
            workload[subset] = KronQuery(domains, subset, factors)

    return workload


def qs_mixed_variance(domains, marginals, mixed_workload, solver_name="fourier",
                      convex_optimizer="lbfgsb_fourier", convex_kwargs=None):
    """Run QS with mixed workload (KronQuery + dense numpy arrays).

    Returns (sum_var, n_queries).
    """
    qs = Smasher(domains, privacy_cost=1.0, default_solver=solver_name,
                 noise=True, verbose=False, convex_optimizer=convex_optimizer,
                 convex_kwargs=convex_kwargs)
    n_queries = 0
    for subset in marginals:
        entry = mixed_workload[subset]
        if isinstance(entry, KronQuery):
            qs.add_queries_to_workload([entry])
            n_q = 1
            for f in entry.factors:
                n_q *= f.shape[0]
            n_queries += n_q
        else:
            # Dense numpy array
            qs.add_queries_to_workload([MatrixQuery(domains, subset, entry)])
            n_queries += entry.shape[0]
    qs.optimize()
    return qs.workload_variance(), n_queries


def build_kron_mixed_basis_workload(domains, marginals, num_attrs):
    """Prefix basis for numerical attributes, identity for categorical.

    Same semantics as build_mixed_basis_workload but returns KronQuery objects
    instead of dense MatrixQuery. This avoids materializing huge Kronecker
    products, enabling 3-way+ marginals on large domains.

    Args:
        domains: list of domain sizes
        marginals: list of attribute subsets
        num_attrs: number of numerical attributes (first num_attrs use prefix)

    Returns:
        dict mapping subset -> KronQuery
    """
    workload = {}
    for subset in marginals:
        factors = []
        for i in subset:
            d = domains[i]
            if i < num_attrs:
                factors.append(prefix_queries(d))
            else:
                factors.append(np.eye(d))
        workload[subset] = KronQuery(domains, subset, factors)
    return workload


def qs_kron_variance(domains, marginals, kron_workload, solver_name="fourier",
                     convex_optimizer="lbfgsb_fourier", convex_kwargs=None):
    """Run QS with KronQuery workload through the Smasher pipeline.

    Args:
        domains: list of domain sizes
        marginals: list of attribute subsets
        kron_workload: dict mapping subset -> KronQuery
        solver_name: "fourier" or "convex"
        convex_optimizer: optimizer for convex solver
        convex_kwargs: extra kwargs for convex solver

    Returns:
        (sum_var, n_queries): total variance and number of queries
    """
    qs = Smasher(domains, privacy_cost=1.0, default_solver=solver_name,
                 noise=True, verbose=False, convex_optimizer=convex_optimizer,
                 convex_kwargs=convex_kwargs)
    for subset in marginals:
        q = kron_workload[subset]
        qs.add_queries_to_workload([q])
    qs.optimize()
    var = qs.workload_variance()
    # Count total queries: prod of factor row counts
    n_queries = 0
    for subset in marginals:
        q = kron_workload[subset]
        n_q = 1
        for f in q.factors:
            n_q *= f.shape[0]
        n_queries += n_q
    return var, n_queries
