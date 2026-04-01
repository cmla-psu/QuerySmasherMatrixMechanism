"""Fill missing d=20,40 rows for tab:scaling-nd-timing (prefix [n]^d, <=2-way).
Runs median of 3 trials for timing. Variance is deterministic."""

import time
import sys
import os
import resource
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exp_utils import k_way_marginals, prefix_queries, Smasher, KronQuery
from query import MatrixQuery


def _build_and_run_kron(domains, marginals, factor_mat, solver_name):
    qs = Smasher(domains, privacy_cost=1.0,
                 default_solver=solver_name,
                 noise=True, verbose=False)
    total_q = 0
    for subset in marginals:
        if len(subset) == 1:
            qs.add_queries_to_workload(
                [MatrixQuery(domains, subset, factor_mat)])
            total_q += factor_mat.shape[0]
        else:
            factors = [factor_mat] * len(subset)
            kq = KronQuery(domains, subset, factors)
            qs.add_queries_to_workload([kq])
            total_q += factor_mat.shape[0] ** len(subset)
    qs.optimize()
    mem_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    return qs.workload_variance(), qs.phase_times, total_q, mem_mb


N_VALUES = [10, 20, 30, 40]
D_FILL = [10, 20, 30, 40, 50]  # all d values for consistency
N_TRIALS = 3

print("Prefix [n]^d timing — fill d=20,40 (median of 3 trials)")
print(f"{'n':>3s} {'d':>3s} {'#Q':>10s} {'WFF':>14s} {'QS':>14s} {'Impr%':>7s} "
      f"{'Decomp':>7s} {'Solve':>7s} {'Assem':>7s} {'Total':>7s} {'Mem(GB)':>8s}")
print("-" * 100)

for n in N_VALUES:
    factor_mat = prefix_queries(n)
    for d in D_FILL:
        domains = [n] * d
        marginals = k_way_marginals(domains, 2)

        # WFF: 1 trial (deterministic variance)
        wff_var, _, n_queries, _ = _build_and_run_kron(
            domains, marginals, factor_mat, "fourier")

        # QS: 3 trials, take median timing
        trials_decomp = []
        trials_solve = []
        trials_assem = []
        qs_var = None
        peak_mem = 0
        for t in range(N_TRIALS):
            var, phases, _, mem = _build_and_run_kron(
                domains, marginals, factor_mat, "convex")
            qs_var = var
            trials_decomp.append(phases["decompose"])
            trials_solve.append(phases["solve"])
            trials_assem.append(phases["assemble"])
            peak_mem = max(peak_mem, mem)

        t_d = np.median(trials_decomp)
        t_s = np.median(trials_solve)
        t_a = np.median(trials_assem)
        t_total = t_d + t_s + t_a
        impr = (1 - qs_var / wff_var) * 100 if wff_var > 0 else 0.0
        mem_gb = peak_mem / 1024

        print(f"{n:3d} {d:3d} {n_queries:10d} {wff_var:14.2f} {qs_var:14.2f} {impr:6.1f}% "
              f"{t_d:6.1f}s {t_s:6.1f}s {t_a:6.1f}s {t_total:6.1f}s {mem_gb:7.1f}")
        sys.stdout.flush()
    print("-" * 100)

print("Done.")
