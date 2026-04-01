#!/usr/bin/env python
"""Fill RP+ and HDMM baseline values for tab:2way-varying-n.

RP+: workload-aware noise allocation via rpp_variance_general().
     Optimizes noise FOR affine/abs_diff, not just evaluates prefix-optimized noise.
HDMM: run with AllRange workload for range queries on [n]^2.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp_utils import (rpp_variance, rpp_variance_general,
                        affine_queries_2d, abs_diff_queries_2d,
                        prefix_queries, range_queries_1d,
                        k_way_marginals, _ensure_rpp)
import numpy as np


def run_rpp_on_workload_type(n, workload_type):
    """Evaluate RP+ on a specific workload type for [n]^2.

    Uses rpp_variance_general() which computes explicit projections
    for noise allocation (handles non-Kronecker workloads correctly).

    Workload includes all <=2-way marginals:
      1-way: prefix queries
      2-way: workload_type queries
    """
    domains = [n, n]
    marginals = k_way_marginals(domains, 2)
    workload = {}
    n_queries = 0
    for subset in marginals:
        if len(subset) == 1:
            W = prefix_queries(domains[subset[0]])
            workload[subset] = W
            n_queries += W.shape[0]
        else:
            d1, d2 = domains[subset[0]], domains[subset[1]]
            if workload_type == "affine":
                W = affine_queries_2d(d1, d2)
            elif workload_type == "abs_diff":
                W = abs_diff_queries_2d(d1, d2)
            elif workload_type == "prefix":
                W = np.kron(prefix_queries(d1), prefix_queries(d2))
            elif workload_type == "range":
                W = np.kron(range_queries_1d(d1), range_queries_1d(d2))
            else:
                raise ValueError(f"Unknown workload type: {workload_type}")
            workload[subset] = W
            n_queries += W.shape[0]

    total_var = rpp_variance_general(domains, marginals, workload,
                                      bases=["P", "P"])
    rmse = np.sqrt(total_var / n_queries)
    return rmse, n_queries


def run_rpp_experiments():
    """Evaluate RP+ on all workload types for [n]^2, n=10..50."""
    print("=" * 70)
    print("RP+ with workload-aware noise allocation on [n]^2")
    print("=" * 70)

    # Sanity check: rpp_variance_general matches rpp_variance on prefix
    domains = [10, 10]
    marginals = k_way_marginals(domains, 2)
    wl_prefix = {}
    for s in marginals:
        if len(s) == 1:
            wl_prefix[s] = prefix_queries(domains[s[0]])
        else:
            wl_prefix[s] = np.kron(prefix_queries(10), prefix_queries(10))
    v1 = rpp_variance(domains, marginals, wl_prefix, bases=["P", "P"])
    v2 = rpp_variance_general(domains, marginals, wl_prefix, bases=["P", "P"])
    print(f"Sanity check (n=10 prefix): rpp_variance={v1:.4f}, "
          f"general={v2:.4f}, diff={abs(v1-v2)/v1*100:.2f}%")
    assert abs(v1 - v2) / v1 < 0.02, f"MISMATCH: {v1} vs {v2}"

    N_VALUES = [10, 20, 30, 40, 50]
    WORKLOADS = ["prefix", "range", "affine", "abs_diff"]

    print(f"\n{'n':>4s} {'workload':>12s} {'#Q':>8s} {'RP+ RMSE':>10s}")
    print("-" * 40)
    for n in N_VALUES:
        for wl in WORKLOADS:
            try:
                rmse, nq = run_rpp_on_workload_type(n, wl)
                print(f"{n:4d} {wl:>12s} {nq:8d} {rmse:10.2f}")
            except Exception as e:
                print(f"{n:4d} {wl:>12s}     FAIL  {e}")
        print("-" * 40)


def run_hdmm_range():
    """Run HDMM with range queries on [n]^2, n=10..50."""
    _ensure_rpp()
    # HDMM is an external dependency — install from https://github.com/dpcomp-org/hdmm
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))),
        'baselines', 'hdmm', 'src'))
    from hdmm import workload, templates

    print("\n" + "=" * 70)
    print("HDMM with range queries on [n]^2")
    print("=" * 70)

    N_VALUES = [10, 20, 30, 40, 50]
    print(f"{'n':>4s} {'#Q':>8s} {'HDMM SumVar':>12s} {'HDMM RMSE':>10s}")
    print("-" * 40)

    for n in N_VALUES:
        ns = (n, n)
        # Build <=2-way range workload
        blocks = []
        # 1-way marginals: AllRange on each attribute
        for i in range(2):
            factors = [workload.Total(ns[j]) if j != i
                       else workload.AllRange(ns[i]) for j in range(2)]
            blocks.append(workload.Kronecker(factors))
        # 2-way marginal: Kron(AllRange, AllRange)
        blocks.append(workload.Kronecker([workload.AllRange(n),
                                          workload.AllRange(n)]))
        W = workload.VStack(blocks)

        # Count queries
        n_queries = 0
        for i in range(2):
            n_queries += ns[i] * (ns[i] + 1) // 2
        n_queries += (n * (n + 1) // 2) ** 2

        try:
            t = templates.DefaultKron(ns, True)
            loss = t.optimize(W)
            rmse = np.sqrt(loss / n_queries)
            print(f"{n:4d} {n_queries:8d} {loss:12.2f} {rmse:10.2f}")
        except Exception as e:
            print(f"{n:4d} {n_queries:8d}       FAIL  {e}")

    print("-" * 40)


if __name__ == "__main__":
    run_rpp_experiments()
    run_hdmm_range()
