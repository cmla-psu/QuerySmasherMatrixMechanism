"""Real datasets with affine/abs_diff 2-way queries for tab:prefix-workloads_b.

Per advisor: show 2 rows per dataset (affine 2-way, abs_diff 2-way).
Only ≤2-way marginals. For pairs with r>5000, fall back to kron(prefix/identity).
Run r≤5000 first, then r≤10000 later.

Usage:
    cd exp && ../.conda/bin/python -u paper_real_affine_abs.py
"""
import sys, os, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from exp_utils import (DATASETS, affine_queries_2d, abs_diff_queries_2d,
                       prefix_queries, k_way_marginals,
                       Smasher, MatrixQuery, KronQuery)

R_THRESHOLD = 5000  # skip dense for pairs with r > this


def run_real_2way(name, query_type, r_threshold):
    ds = DATASETS[name]
    domains = ds['domains']
    num_attrs = ds['num_attrs']
    marginals = k_way_marginals(domains, 2)  # only ≤2-way

    for solver_name, label in [('fourier', 'WFF'), ('convex', 'QS')]:
        t0 = time.time()
        qs = Smasher(domains, privacy_cost=1.0, default_solver=solver_name,
                     noise=True, verbose=False)
        total_q = 0
        n_dense = 0
        n_kron_fallback = 0

        for subset in marginals:
            k = len(subset)
            sub_domains = [domains[i] for i in subset]

            if k == 1:
                d = sub_domains[0]
                W = prefix_queries(d) if subset[0] < num_attrs else np.eye(d)
                qs.add_queries_to_workload([MatrixQuery(domains, subset, W)])
                total_q += W.shape[0]
            elif k == 2:
                d1, d2 = sub_domains
                r = (d1 - 1) * (d2 - 1)
                if r <= r_threshold:
                    # Dense affine or abs_diff
                    if query_type == 'affine':
                        W = affine_queries_2d(d1, d2)
                    else:
                        W = abs_diff_queries_2d(d1, d2)
                    qs.add_queries_to_workload([MatrixQuery(domains, subset, W)])
                    total_q += W.shape[0]
                    n_dense += 1
                else:
                    # Kron fallback: prefix/identity per attribute
                    factors = []
                    for i in subset:
                        d = domains[i]
                        factors.append(prefix_queries(d) if i < num_attrs else np.eye(d))
                    kq = KronQuery(domains, subset, factors)
                    qs.add_queries_to_workload([kq])
                    nq = 1
                    for f in factors:
                        nq *= f.shape[0]
                    total_q += nq
                    n_kron_fallback += 1

        qs.optimize()
        var = qs.workload_variance()
        rmse = np.sqrt(var / total_q)
        elapsed = time.time() - t0

        if label == 'WFF':
            wff_rmse = rmse
        else:
            qs_rmse = rmse
            impr = (1 - qs_rmse / wff_rmse) * 100
            print(f'  {name:>6s} {query_type:>8s}  #Q={total_q:>10,d}  '
                  f'WFF={wff_rmse:8.3f}  QS={qs_rmse:8.3f}  impr={impr:5.1f}%  '
                  f'time={elapsed:.1f}s  dense={n_dense} kron={n_kron_fallback}')
    sys.stdout.flush()


print(f"Real datasets: affine/abs_diff 2-way (r≤{R_THRESHOLD} dense, rest kron fallback)")
print(f"Only ≤2-way marginals.")
print()

for name in ['CPS', 'Adult', 'Loans']:
    ds = DATASETS[name]
    domains = ds['domains']
    # Show r distribution
    marginals_2way = [(i,j) for i in range(len(domains)) for j in range(i+1,len(domains))]
    n_under = sum(1 for i,j in marginals_2way if (domains[i]-1)*(domains[j]-1) <= R_THRESHOLD)
    n_over = len(marginals_2way) - n_under
    print(f'{name}: {len(marginals_2way)} 2-way pairs, {n_under} with r≤{R_THRESHOLD}, {n_over} with r>{R_THRESHOLD}')

    for qt in ['affine', 'abs_diff']:
        run_real_2way(name, qt, R_THRESHOLD)
    print()

print("Done.")
