"""Fill missing unweighted rows: n=40 (affine/abs_diff/random) + n=50 (all except circ_range).
Skips circ_range (WFF=QS, Fourier-optimal). Uses JAX CPU for convex solver."""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_utils import (
    k_way_marginals,
    prefix_queries, range_queries_1d, circular_range_queries,
    affine_queries_2d, abs_diff_queries_2d,
    Smasher, MatrixQuery, KronQuery, _ensure_rpp,
)

D_ATTR = 40

def run_qs_kron(domains, marginals, factor_1way, factors_2way, solver_name, use_jax=False):
    qs = Smasher(domains, privacy_cost=1.0, default_solver=solver_name,
                 noise=True, verbose=False, use_jax=use_jax)
    total_q = 0
    for subset in marginals:
        if len(subset) == 1:
            qs.add_queries_to_workload([MatrixQuery(domains, subset, factor_1way)])
            total_q += factor_1way.shape[0]
        else:
            kq = KronQuery(domains, subset, list(factors_2way))
            qs.add_queries_to_workload([kq])
            nq = 1
            for f in factors_2way:
                nq *= f.shape[0]
            total_q += nq
    qs.optimize()
    return qs.workload_variance(), total_q

def run_qs_dense(domains, marginals, W_1way, W_2way, solver_name, use_jax=False):
    qs = Smasher(domains, privacy_cost=1.0, default_solver=solver_name,
                 noise=True, verbose=False, use_jax=use_jax)
    total_q = 0
    for subset in marginals:
        if len(subset) == 1:
            qs.add_queries_to_workload([MatrixQuery(domains, subset, W_1way)])
            total_q += W_1way.shape[0]
        else:
            qs.add_queries_to_workload([MatrixQuery(domains, subset, W_2way)])
            total_q += W_2way.shape[0]
    qs.optimize()
    return qs.workload_variance(), total_q

def run_hdmm(d, n, wl_cls_name):
    _ensure_rpp()
    # HDMM is an external dependency — install from https://github.com/dpcomp-org/hdmm
    hpath = os.path.join('..', 'baselines', 'hdmm', 'src')
    if hpath not in sys.path:
        sys.path.insert(0, hpath)
    from hdmm import workload as hw, templates
    wl_cls = hw.Prefix if wl_cls_name == 'prefix' else hw.AllRange
    blocks = []
    for i in range(d):
        factors = [hw.Total(n) if j != i else wl_cls(n) for j in range(d)]
        blocks.append(hw.Kronecker(factors))
    for i in range(d):
        for j in range(i+1, d):
            factors = [hw.Total(n) if k not in (i,j) else wl_cls(n) for k in range(d)]
            blocks.append(hw.Kronecker(factors))
    W_h = hw.VStack(blocks)
    return templates.DefaultKron(tuple([n]*d), True).optimize(W_h)

d = D_ATTR
print(f"Filling missing unweighted rows for [n]^{d} (skip circ_range)")
print(f"{'n':>4s} {'workload':>12s} {'#Q':>12s} {'HDMM':>10s} {'WFF':>10s} {'QS':>10s}")
print("-" * 65)

CONFIGS = [
    (40, ['affine', 'abs_diff', 'random']),
    (50, ['marginal', 'prefix', 'range', 'affine', 'abs_diff', 'random']),
]

for n, workloads in CONFIGS:
    domains = [n] * d
    marginals = k_way_marginals(domains, 2)

    for wl_name in workloads:
        hdmm_str = '        NI'
        W_pfx = prefix_queries(n)

        if wl_name == 'marginal':
            W_id = np.eye(n)
            wff_var, nq = run_qs_kron(domains, marginals, W_id, [W_id, W_id], "fourier")
            rmse = np.sqrt(wff_var / nq)
            print(f"{n:4d} {'marginal':>12s} {nq:12,d} {rmse:10.2f} {rmse:10.2f} {rmse:10.2f}")
            sys.stdout.flush()
            continue

        if wl_name == 'prefix':
            P = prefix_queries(n)
            wff_var, nq = run_qs_kron(domains, marginals, P, [P, P], "fourier")
            wff_rmse = np.sqrt(wff_var / nq)
            qs_var, _ = run_qs_kron(domains, marginals, P, [P, P], "convex", use_jax=True)
            qs_rmse = np.sqrt(qs_var / nq)
            hdmm_var = run_hdmm(d, n, 'prefix')
            hdmm_str = f'{np.sqrt(hdmm_var / nq):10.2f}'
        elif wl_name == 'range':
            R = range_queries_1d(n)
            wff_var, nq = run_qs_kron(domains, marginals, R, [R, R], "fourier")
            wff_rmse = np.sqrt(wff_var / nq)
            qs_var, _ = run_qs_kron(domains, marginals, R, [R, R], "convex", use_jax=True)
            qs_rmse = np.sqrt(qs_var / nq)
            hdmm_var = run_hdmm(d, n, 'range')
            hdmm_str = f'{np.sqrt(hdmm_var / nq):10.2f}'
        elif wl_name == 'affine':
            W_aff = affine_queries_2d(n, n)
            wff_var, nq = run_qs_dense(domains, marginals, W_pfx, W_aff, "fourier")
            wff_rmse = np.sqrt(wff_var / nq)
            qs_var, _ = run_qs_dense(domains, marginals, W_pfx, W_aff, "convex", use_jax=True)
            qs_rmse = np.sqrt(qs_var / nq)
        elif wl_name == 'abs_diff':
            W_abs = abs_diff_queries_2d(n, n)
            wff_var, nq = run_qs_dense(domains, marginals, W_pfx, W_abs, "fourier")
            wff_rmse = np.sqrt(wff_var / nq)
            qs_var, _ = run_qs_dense(domains, marginals, W_pfx, W_abs, "convex", use_jax=True)
            qs_rmse = np.sqrt(qs_var / nq)
        elif wl_name == 'random':
            np.random.seed(42 + n)
            W1 = (np.random.rand(3*n, n) < 0.3).astype(float)
            W2 = (np.random.rand(3*n*n, n*n) < 0.3).astype(float)
            wff_var, nq = run_qs_dense(domains, marginals, W1, W2, "fourier")
            wff_rmse = np.sqrt(wff_var / nq)
            qs_var, _ = run_qs_dense(domains, marginals, W1, W2, "convex", use_jax=True)
            qs_rmse = np.sqrt(qs_var / nq)

        row_n = f"{n:4d}" if wl_name == workloads[0] else "    "
        print(f"{row_n} {wl_name:>12s} {nq:12,d} {hdmm_str} {wff_rmse:10.2f} {qs_rmse:10.2f}")
        sys.stdout.flush()

    print("-" * 65)

print("Done.")
