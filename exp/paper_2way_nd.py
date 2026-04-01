"""2-way workloads on [n]^d, varying domain size n.
Replaces tab:2way-varying-n (was [n]^2, now [n]^d with d=D_ATTR).

Runs unweighted experiment first, then weighted version.

Usage:
    cd exp && ../.conda/bin/python -u paper_2way_nd.py
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_utils import (
    k_way_marginals,
    prefix_queries, range_queries_1d, circular_range_queries,
    affine_queries_2d, abs_diff_queries_2d,
    rpp_variance,
    Smasher, MatrixQuery, KronQuery,
    _ensure_rpp,
)

D_ATTR = 40  # number of attributes
N_VALUES = [10, 20, 30, 40, 50]

# RP+ becomes very slow for Kron workloads when n is large.
# rpp_variance computes Cov (n^2 x n^2) per marginal: O(820 * n^6).
# Skip RP+ on Kron workloads when n > this threshold.
RPP_KRON_MAX_N = 20

# RP+ on dense workloads (affine/abs_diff) is lighter (small query matrices).
RPP_DENSE_MAX_N = 30

# HDMM is very slow with d=40 (820 Kronecker blocks).
# Only run for n=10 to get one comparison point.
HDMM_MAX_N = 10


# ======================================================================
# Helpers
# ======================================================================

def run_qs_kron(domains, marginals, factor_1way, factors_2way, solver_name):
    """QS with KronQuery for separable 2-way workloads."""
    qs = Smasher(domains, privacy_cost=1.0, default_solver=solver_name,
                 noise=True, verbose=False)
    total_q = 0
    for subset in marginals:
        if len(subset) == 1:
            qs.add_queries_to_workload(
                [MatrixQuery(domains, subset, factor_1way)])
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


def run_qs_dense(domains, marginals, W_1way, W_2way, solver_name):
    """QS with dense MatrixQuery for non-separable 2-way workloads."""
    qs = Smasher(domains, privacy_cost=1.0, default_solver=solver_name,
                 noise=True, verbose=False)
    total_q = 0
    for subset in marginals:
        if len(subset) == 1:
            qs.add_queries_to_workload(
                [MatrixQuery(domains, subset, W_1way)])
            total_q += W_1way.shape[0]
        else:
            qs.add_queries_to_workload(
                [MatrixQuery(domains, subset, W_2way)])
            total_q += W_2way.shape[0]
    qs.optimize()
    return qs.workload_variance(), total_q


def run_rpp_kron(domains, marginals, factor_1way, factors_2way):
    """RP+ on Kronecker workload. Reuses kron product to save memory."""
    from functools import reduce
    kron_2way = reduce(np.kron, factors_2way)  # compute once
    workload = {}
    for subset in marginals:
        if len(subset) == 1:
            workload[subset] = factor_1way
        else:
            workload[subset] = kron_2way  # same object for all 2-way
    bases = ["P"] * len(domains)
    return rpp_variance(domains, marginals, workload, bases=bases)


def run_rpp_dense(domains, marginals, W_1way, W_2way):
    """RP+ on dense workload (affine/abs_diff)."""
    workload = {}
    for subset in marginals:
        if len(subset) == 1:
            workload[subset] = W_1way
        else:
            workload[subset] = W_2way  # same object for all 2-way
    bases = ["P"] * len(domains)
    from exp_utils import rpp_variance_general
    return rpp_variance_general(domains, marginals, workload, bases=bases)


def run_hdmm(domains, d, n, wl_cls_name):
    """HDMM on prefix/range Kronecker workload."""
    _ensure_rpp()
    # HDMM is an external dependency — install from https://github.com/dpcomp-org/hdmm
    hpath = os.path.join('..', 'baselines', 'hdmm', 'src')
    if hpath not in sys.path:
        sys.path.insert(0, hpath)
    from hdmm import workload as hw, templates
    if wl_cls_name == 'prefix':
        wl_cls = hw.Prefix
    elif wl_cls_name == 'range':
        wl_cls = hw.AllRange
    else:
        return None

    blocks = []
    for i in range(d):
        factors = [hw.Total(n) if j != i else wl_cls(n) for j in range(d)]
        blocks.append(hw.Kronecker(factors))
    for i in range(d):
        for j in range(i + 1, d):
            factors = [hw.Total(n) if k not in (i, j) else wl_cls(n) for k in range(d)]
            blocks.append(hw.Kronecker(factors))
    W_h = hw.VStack(blocks)
    loss = templates.DefaultKron(tuple([n]*d), True).optimize(W_h)
    return loss


def rpp_str(n, max_n, domains, marginals, run_fn, *args):
    """Run RP+ if n <= max_n, else return NI."""
    if n > max_n:
        return '        NI'
    try:
        var = run_fn(domains, marginals, *args)
        nq = args[-1] if isinstance(args[-1], int) else None
        return var
    except Exception as e:
        print(f"    [RP+ failed: {e}]", file=sys.stderr)
        return None


# ======================================================================
# Main experiment
# ======================================================================

def run_experiment():
    d = D_ATTR
    print("=" * 100)
    print(f"  2-way workloads on [n]^{d}, varying n")
    print(f"  RP+ shown for n<={RPP_KRON_MAX_N} (Kron), n<={RPP_DENSE_MAX_N} (dense)")
    print(f"  HDMM shown for n<={HDMM_MAX_N}")
    print("=" * 100)

    print(f"{'n':>4s} {'workload':>12s} {'#Q':>12s} "
          f"{'RP+':>10s} {'HDMM':>10s} {'WFF':>10s} {'QS':>10s}")
    print("-" * 80)

    for n in N_VALUES:
        domains = [n] * d
        marginals = k_way_marginals(domains, 2)

        # --- 1. Marginal (identity) ---
        W_id_1 = np.eye(n)
        wff_var, nq = run_qs_kron(domains, marginals, W_id_1, [W_id_1, W_id_1], "fourier")
        rmse = np.sqrt(wff_var / nq)
        print(f"{n:4d} {'marginal':>12s} {nq:12,d} {rmse:10.4f} {rmse:10.4f} {rmse:10.4f} {rmse:10.4f}")
        sys.stdout.flush()

        # --- 2. Prefix (Kron) ---
        P = prefix_queries(n)
        wff_var, nq = run_qs_kron(domains, marginals, P, [P, P], "fourier")
        wff_rmse = np.sqrt(wff_var / nq)
        qs_var, _ = run_qs_kron(domains, marginals, P, [P, P], "convex")
        qs_rmse = np.sqrt(qs_var / nq)

        rpp_rmse_str = '        NI'
        if n <= RPP_KRON_MAX_N:
            try:
                rpp_var = run_rpp_kron(domains, marginals, P, [P, P])
                rpp_rmse_str = f'{np.sqrt(rpp_var / nq):10.2f}'
            except Exception as e:
                print(f"    [RP+ prefix n={n} failed: {e}]", file=sys.stderr)

        hdmm_rmse_str = '        NI'
        if n <= HDMM_MAX_N:
            try:
                hdmm_var = run_hdmm(domains, d, n, 'prefix')
                if hdmm_var is not None:
                    hdmm_rmse_str = f'{np.sqrt(hdmm_var / nq):10.2f}'
            except Exception as e:
                print(f"    [HDMM prefix n={n} failed: {e}]", file=sys.stderr)

        print(f"     {'prefix':>12s} {nq:12,d} {rpp_rmse_str} {hdmm_rmse_str} {wff_rmse:10.2f} {qs_rmse:10.2f}")
        sys.stdout.flush()

        # --- 3. Range (Kron) ---
        R = range_queries_1d(n)
        wff_var, nq = run_qs_kron(domains, marginals, R, [R, R], "fourier")
        wff_rmse = np.sqrt(wff_var / nq)
        qs_var, _ = run_qs_kron(domains, marginals, R, [R, R], "convex")
        qs_rmse = np.sqrt(qs_var / nq)

        rpp_rmse_str = '        NI'
        if n <= RPP_KRON_MAX_N:
            try:
                rpp_var = run_rpp_kron(domains, marginals, R, [R, R])
                rpp_rmse_str = f'{np.sqrt(rpp_var / nq):10.2f}'
            except Exception as e:
                print(f"    [RP+ range n={n} failed: {e}]", file=sys.stderr)

        hdmm_rmse_str = '        NI'
        if n <= HDMM_MAX_N:
            try:
                hdmm_var = run_hdmm(domains, d, n, 'range')
                if hdmm_var is not None:
                    hdmm_rmse_str = f'{np.sqrt(hdmm_var / nq):10.2f}'
            except Exception as e:
                print(f"    [HDMM range n={n} failed: {e}]", file=sys.stderr)

        print(f"     {'range':>12s} {nq:12,d} {rpp_rmse_str} {hdmm_rmse_str} {wff_rmse:10.2f} {qs_rmse:10.2f}")
        sys.stdout.flush()

        # --- 4. Circ_range (Kron) ---
        C = circular_range_queries(n)
        wff_var, nq = run_qs_kron(domains, marginals, C, [C, C], "fourier")
        wff_rmse = np.sqrt(wff_var / nq)
        qs_var, _ = run_qs_kron(domains, marginals, C, [C, C], "convex")
        qs_rmse = np.sqrt(qs_var / nq)
        print(f"     {'circ_range':>12s} {nq:12,d} {'        NI'} {'        NI'} {wff_rmse:10.2f} {qs_rmse:10.2f}")
        sys.stdout.flush()

        # --- 5. Affine (dense 2-way) ---
        W_aff = affine_queries_2d(n, n)
        W_pfx = prefix_queries(n)
        wff_var, nq = run_qs_dense(domains, marginals, W_pfx, W_aff, "fourier")
        wff_rmse = np.sqrt(wff_var / nq)
        qs_var, _ = run_qs_dense(domains, marginals, W_pfx, W_aff, "convex")
        qs_rmse = np.sqrt(qs_var / nq)

        rpp_rmse_str = '        NI'
        if n <= RPP_DENSE_MAX_N:
            try:
                rpp_var = run_rpp_dense(domains, marginals, W_pfx, W_aff)
                rpp_rmse_str = f'{np.sqrt(rpp_var / nq):10.2f}'
            except Exception as e:
                print(f"    [RP+ affine n={n} failed: {e}]", file=sys.stderr)

        print(f"     {'affine':>12s} {nq:12,d} {rpp_rmse_str} {'        NI'} {wff_rmse:10.2f} {qs_rmse:10.2f}")
        sys.stdout.flush()

        # --- 6. Abs_diff (dense 2-way) ---
        W_abs = abs_diff_queries_2d(n, n)
        wff_var, nq = run_qs_dense(domains, marginals, W_pfx, W_abs, "fourier")
        wff_rmse = np.sqrt(wff_var / nq)
        qs_var, _ = run_qs_dense(domains, marginals, W_pfx, W_abs, "convex")
        qs_rmse = np.sqrt(qs_var / nq)

        rpp_rmse_str = '        NI'
        if n <= RPP_DENSE_MAX_N:
            try:
                rpp_var = run_rpp_dense(domains, marginals, W_pfx, W_abs)
                rpp_rmse_str = f'{np.sqrt(rpp_var / nq):10.2f}'
            except Exception as e:
                print(f"    [RP+ abs_diff n={n} failed: {e}]", file=sys.stderr)

        print(f"     {'abs_diff':>12s} {nq:12,d} {rpp_rmse_str} {'        NI'} {wff_rmse:10.2f} {qs_rmse:10.2f}")
        sys.stdout.flush()

        # --- 7. Random (dense 2-way) ---
        np.random.seed(42 + n)
        W_rand_1 = (np.random.rand(3 * n, n) < 0.3).astype(float)
        W_rand_2 = (np.random.rand(3 * n * n, n * n) < 0.3).astype(float)
        wff_var, nq = run_qs_dense(domains, marginals, W_rand_1, W_rand_2, "fourier")
        wff_rmse = np.sqrt(wff_var / nq)
        qs_var, _ = run_qs_dense(domains, marginals, W_rand_1, W_rand_2, "convex")
        qs_rmse = np.sqrt(qs_var / nq)
        print(f"     {'random':>12s} {nq:12,d} {'        NI'} {'        NI'} {wff_rmse:10.2f} {qs_rmse:10.2f}")
        sys.stdout.flush()

        print("-" * 80)

    print()


def run_weighted(weight_seed=42):
    """Weighted version: per-marginal weights from {1,...,5}."""
    d = D_ATTR
    print("=" * 80)
    print(f"  Weighted 2-way workloads on [n]^{d}, varying n")
    print(f"  Per-marginal weights from {{1,...,5}}, seed={weight_seed}")
    print("=" * 80)

    print(f"{'n':>4s} {'workload':>12s} {'#Q':>12s} {'WFF':>10s} {'QS':>10s}")
    print("-" * 55)

    for n in N_VALUES:
        domains = [n] * d
        marginals = k_way_marginals(domains, 2)

        # === Kron workloads ===
        KRON_WLS = [
            ('marginal',   np.eye(n),                [np.eye(n)] * 2),
            ('prefix',     prefix_queries(n),         [prefix_queries(n)] * 2),
            ('range',      range_queries_1d(n),       [range_queries_1d(n)] * 2),
            ('circ_range', circular_range_queries(n),  [circular_range_queries(n)] * 2),
        ]

        for wl_name, factor_1, factors_2 in KRON_WLS:
            nq_1 = factor_1.shape[0]
            nq_2 = 1
            for f in factors_2:
                nq_2 *= f.shape[0]
            total_q = d * nq_1 + d * (d - 1) // 2 * nq_2

            wff_rmse = qs_rmse = 0.0
            for solver_name in ["fourier", "convex"]:
                qs = Smasher(domains, privacy_cost=1.0, default_solver=solver_name,
                             noise=True, verbose=False)
                rng_local = np.random.RandomState(weight_seed)
                for subset in marginals:
                    w = float(rng_local.randint(1, 6))
                    if len(subset) == 1:
                        qs.add_queries_to_workload(
                            [MatrixQuery(domains, subset, factor_1, weight=w)])
                    else:
                        kq = KronQuery(domains, subset, list(factors_2), weight=w)
                        qs.add_queries_to_workload([kq])
                qs.optimize()
                var = qs.workload_variance()
                rmse = np.sqrt(var / total_q)
                if solver_name == "fourier":
                    wff_rmse = rmse
                else:
                    qs_rmse = rmse

            row_n = f"{n:4d}" if wl_name == KRON_WLS[0][0] else "    "
            print(f"{row_n} {wl_name:>12s} {total_q:12,d} {wff_rmse:10.3f} {qs_rmse:10.3f}")
            sys.stdout.flush()

        # === Dense workloads ===
        W_pfx = prefix_queries(n)
        for wl_name, W_2way in [('affine', affine_queries_2d(n, n)),
                                 ('abs_diff', abs_diff_queries_2d(n, n))]:
            nq_1 = W_pfx.shape[0]
            nq_2 = W_2way.shape[0]
            total_q = d * nq_1 + d * (d - 1) // 2 * nq_2

            wff_rmse = qs_rmse = 0.0
            for solver_name in ["fourier", "convex"]:
                qs = Smasher(domains, privacy_cost=1.0, default_solver=solver_name,
                             noise=True, verbose=False)
                rng_local = np.random.RandomState(weight_seed)
                for subset in marginals:
                    w = float(rng_local.randint(1, 6))
                    if len(subset) == 1:
                        qs.add_queries_to_workload(
                            [MatrixQuery(domains, subset, W_pfx, weight=w)])
                    else:
                        qs.add_queries_to_workload(
                            [MatrixQuery(domains, subset, W_2way, weight=w)])
                qs.optimize()
                var = qs.workload_variance()
                rmse = np.sqrt(var / total_q)
                if solver_name == "fourier":
                    wff_rmse = rmse
                else:
                    qs_rmse = rmse

            print(f"     {wl_name:>12s} {total_q:12,d} {wff_rmse:10.3f} {qs_rmse:10.3f}")
            sys.stdout.flush()

        # Random (weighted)
        np.random.seed(42 + n)
        W_rand_1 = (np.random.rand(3 * n, n) < 0.3).astype(float)
        W_rand_2 = (np.random.rand(3 * n * n, n * n) < 0.3).astype(float)
        nq_1 = W_rand_1.shape[0]
        nq_2 = W_rand_2.shape[0]
        total_q = d * nq_1 + d * (d - 1) // 2 * nq_2

        wff_rmse = qs_rmse = 0.0
        for solver_name in ["fourier", "convex"]:
            qs = Smasher(domains, privacy_cost=1.0, default_solver=solver_name,
                         noise=True, verbose=False)
            rng_local = np.random.RandomState(weight_seed)
            for subset in marginals:
                w = float(rng_local.randint(1, 6))
                if len(subset) == 1:
                    qs.add_queries_to_workload(
                        [MatrixQuery(domains, subset, W_rand_1, weight=w)])
                else:
                    qs.add_queries_to_workload(
                        [MatrixQuery(domains, subset, W_rand_2, weight=w)])
            qs.optimize()
            var = qs.workload_variance()
            rmse = np.sqrt(var / total_q)
            if solver_name == "fourier":
                wff_rmse = rmse
            else:
                qs_rmse = rmse

        print(f"     {'random':>12s} {total_q:12,d} {wff_rmse:10.3f} {qs_rmse:10.3f}")
        sys.stdout.flush()
        print("-" * 55)

    print()


if __name__ == '__main__':
    print(f"d = {D_ATTR} attributes, n varies in {N_VALUES}")
    n_marg = D_ATTR + D_ATTR * (D_ATTR - 1) // 2
    print(f"Marginals: ≤2-way → {D_ATTR} 1-way + {D_ATTR*(D_ATTR-1)//2} 2-way = {n_marg} total")
    print()

    run_experiment()
    run_weighted()

    print("All experiments complete.")
