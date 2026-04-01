#!/usr/bin/env python
"""Mixed-query scaling on [n]^d with ≤3-way marginals.

1-way = range, 2-way = affine (dense), 3-way = prefix (Kron).
Varies both n and d, reporting accuracy and timing.

Produces data for:
  - tab:mixed-scaling-d-accuracy (RMSE, n×d grid)
  - tab:mixed-scaling-d-timing  (phase times, n×d grid)
"""
import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exp_utils import (build_mixed_workload_v2, qs_mixed_variance,
                        k_way_marginals)
from query import MatrixQuery, KronQuery
from smasher import Smasher
import numpy as np


N_VALUES = [10, 20, 30, 40]
D_VALUES = [10, 20, 30, 40, 50]


def count_queries_mixed(mixed_wl, marginals):
    nq = 0
    for subset in marginals:
        entry = mixed_wl[subset]
        if isinstance(entry, KronQuery):
            n_q = 1
            for f in entry.factors:
                n_q *= f.shape[0]
            nq += n_q
        else:
            nq += entry.shape[0]
    return nq


def run_mixed_nd():
    print("=" * 100)
    print("Mixed-query scaling: [n]^d, ≤3-way marginals")
    print("1-way=range, 2-way=affine, 3-way=prefix (Kron)")
    print("=" * 100)
    print(f"{'n':>4s} {'d':>4s} {'#Queries':>12s} {'WFF':>10s} {'QS':>10s} "
          f"{'Impr%':>7s} {'Decomp':>8s} {'Solve':>8s} {'Assem':>8s} {'Total':>8s}")
    print("-" * 100)

    for n in N_VALUES:
        for d in D_VALUES:
            domains = [n] * d
            marginals = k_way_marginals(domains, 3)
            mixed_wl = build_mixed_workload_v2(domains, marginals)
            n_queries = count_queries_mixed(mixed_wl, marginals)

            # --- WFF (Fourier) ---
            wff_var, _ = qs_mixed_variance(
                domains, marginals, mixed_wl, solver_name="fourier")
            wff_rmse = np.sqrt(wff_var / n_queries)

            # --- QS (Convex) — use optimize() + phase_times ---
            qs = Smasher(domains, privacy_cost=1.0, default_solver="convex",
                         noise=True, verbose=False,
                         convex_optimizer="lbfgsb_fourier")
            for subset in marginals:
                entry = mixed_wl[subset]
                if isinstance(entry, KronQuery):
                    qs.add_queries_to_workload([entry])
                else:
                    qs.add_queries_to_workload(
                        [MatrixQuery(domains, subset, entry)])

            qs.optimize()
            qs_var = qs.workload_variance()
            qs_rmse = np.sqrt(qs_var / n_queries)
            impr = (1 - qs_rmse / wff_rmse) * 100

            phases = qs.phase_times
            t_d = phases.get("decompose", 0)
            t_s = phases.get("solve", 0)
            t_a = phases.get("assemble", 0)
            total = t_d + t_s + t_a

            print(f"{n:4d} {d:4d} {n_queries:12,d} {wff_rmse:10.2f} {qs_rmse:10.2f} "
                  f"{impr:6.1f}% {t_d:7.1f}s {t_s:7.1f}s "
                  f"{t_a:7.1f}s {total:7.1f}s")
            sys.stdout.flush()
        print("-" * 100)
        sys.stdout.flush()


if __name__ == "__main__":
    run_mixed_nd()
