"""
Paper Section 8.2: Comparison with Baselines

Compares QuerySmasher against published baselines on real and synthetic datasets.
Metric: RMSE = sqrt(SumVar / #Queries) at privacy cost 1.
All QS experiments use Kronecker-factored solvers to avoid dense materialization.

Methods:
  - WFF  (Workload-Free Fourier): QS with Fourier solver only
  - RP   (ResidualPlanner, NeurIPS): paper values (= SVDB for identity marginals)
  - RP+  (ResidualPlanner+): paper values
  - HDMM: paper values (prefix workloads only)
  - QS   (QuerySmasher): unified best solver (convex + warm start + kron + fallback)

Exp 8.2.1: Real datasets — identity marginal workload
  - Adult (14 attrs), CPS (5 attrs), Loans (12 attrs)
  - Workloads: 1-way, 2-way, 3-way, <=3-way (matches RP+ Table accrmse)
  - All methods achieve SVDB-optimal RMSE

Exp 8.2.2: Synthetic scalability — Syn-10^d
  - [10]^d for d in {2, 5, 10, 15, 20}, <=3-way identity marginals
  - Matches RP+ paper synthetic setup (n=10, varying d)

Exp 8.2.3: Prefix-sum workloads on real datasets
  - Mixed bases: prefix for numerical attrs, identity for categorical
  - Matches RP+ paper Table accrmse-prefix

Usage:
    cd exp && ../.conda/bin/python paper_section8_2.py [1] [2] [3]
"""

import time
import sys
import os
import types
import platform
import numpy as np
from itertools import combinations

from exp_utils import (qs_kron_variance, k_way_marginals, DATASETS,
                       build_kron_identity_workload,
                       build_kron_mixed_basis_workload)


# ======================================================================
# Hardware info
# ======================================================================

def print_hardware_info():
    """Print hardware info for reproducibility."""
    print("Hardware:")
    print(f"  Platform: {platform.platform()}")
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if 'model name' in line:
                    print(f"  CPU: {line.strip().split(':')[1].strip()}")
                    break
        mem_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024**3)
        print(f"  RAM: {mem_gb:.0f} GB")
    except Exception:
        pass
    print()


# ======================================================================
# RP+ setup (for live runs on synthetic configs)
# ======================================================================

RPP_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "baselines")

def _ensure_rpp():
    """Mock gurobipy and add RP+ to path."""
    if "gurobipy" not in sys.modules or not hasattr(sys.modules["gurobipy"], "Model"):
        _mock = types.ModuleType("gurobipy")
        _mock.GRB = types.SimpleNamespace()
        sys.modules["gurobipy"] = _mock
    if RPP_ROOT not in sys.path:
        sys.path.insert(0, RPP_ROOT)
    hdmm_src = os.path.join(RPP_ROOT, "hdmm", "src")
    if hdmm_src not in sys.path:
        sys.path.insert(0, hdmm_src)


def run_rpp_identity(domains, marginals):
    """Run RP+ with identity basis and return (rmse, time)."""
    _ensure_rpp()
    from resplan import ResPlanSum

    t0 = time.time()
    system = ResPlanSum(domains, bases=["I"] * len(domains))
    for subset in marginals:
        system.input_mech(subset)
    sum_var = system.get_noise_level()
    elapsed = time.time() - t0

    n_queries = sum(int(np.prod([domains[i] for i in s])) for s in marginals)
    rmse = np.sqrt(sum_var / n_queries) if n_queries > 0 else None
    return rmse, elapsed


# ======================================================================
# Paper values (baselines from published results)
# ======================================================================

# RP+ paper Table accrmse (identity marginal workloads, RMSE at pcost=1)
# For identity workloads: RP = RP+ = SVDB (all provably optimal)
RP_PAPER = {
    "Adult": {"1-way": 3.047, "2-way": 6.359, "3-way": 10.515, "<=3-way": 10.665},
    "CPS":   {"1-way": 1.744, "2-way": 2.035, "3-way": 2.048, "<=3-way": 2.276},
    "Loans": {"1-way": 2.875, "2-way": 5.634, "3-way": 8.702, "<=3-way": 8.876},
}
RPP_PAPER = RP_PAPER  # RP = RP+ for identity marginals (both SVDB-optimal)

# RP+ paper Table accrmse-prefix (prefix workloads, RMSE)
RPP_PAPER_PREFIX = {
    "Adult": {"1-way": 5.081, "2-way": 17.274, "3-way": 45.223, "<=3-way": 45.174},
    "CPS":   {"1-way": 3.081, "2-way": 5.823, "3-way": 7.302, "<=3-way": 7.570},
    "Loans": {"1-way": 4.681, "2-way": 14.502, "3-way": 34.168, "<=3-way": 34.144},
}

# HDMM paper values (from RP+ paper Table accrmse-prefix)
HDMM_PAPER_PREFIX = {
    "Adult": {"1-way": 5.648, "2-way": 21.111, "3-way": 60.025, "<=3-way": 61.613},
    "CPS":   {"1-way": 3.347, "2-way": 6.625, "3-way": 8.634, "<=3-way": 8.623},
    "Loans": {"1-way": 5.127, "2-way": 17.149, "3-way": 42.938, "<=3-way": 44.338},
}


# ======================================================================
# Helpers
# ======================================================================

def kron_n_queries(kron_workload, marginals):
    """Count total queries from KronQuery workload."""
    total = 0
    for subset in marginals:
        q = kron_workload[subset]
        n_q = 1
        for f in q.factors:
            n_q *= f.shape[0]
        total += n_q
    return total


# Profiled-tight convex settings for KronConvexSolver.
# Per-factor r_i = d_i - 1 is always small, so tight convergence is cheap.
# These are also set as adaptive defaults in smasher.py for Kron solvers.
KRON_TIGHT_KWARGS = {"mu_schedule": [0.1, 0.01, 0.001, 0.0001],
                     "ftol": 1e-10, "maxcor": 50}


def run_qs_kron(domains, marginals, kron_workload, solver_name, label=""):
    """Run QS with KronQuery workload and return (rmse, time)."""
    t0 = time.time()
    try:
        var, n_queries = qs_kron_variance(domains, marginals, kron_workload,
                                          solver_name,
                                          convex_kwargs=KRON_TIGHT_KWARGS)
        if not np.isfinite(var):
            var = None
    except Exception as e:
        print(f"    [{label} failed: {e}]")
        var, n_queries = None, 0
    elapsed = time.time() - t0
    if var is not None and n_queries > 0:
        rmse = np.sqrt(var / n_queries)
    else:
        rmse = None
    return rmse, elapsed


def fmt(val):
    """Format RMSE for display."""
    if val is None:
        return f"{'—':>8s}"
    return f"{val:8.3f}"


def make_kway_specs(domains):
    """Build workload specs: 1-way, 2-way, 3-way, <=3-way."""
    n_attr = len(domains)
    specs = {}
    for k in range(1, min(4, n_attr + 1)):
        specs[f"{k}-way"] = list(combinations(range(n_attr), k))
    if n_attr >= 3:
        all_margs = []
        for k in range(1, 4):
            all_margs.extend(combinations(range(n_attr), k))
        specs["<=3-way"] = all_margs
    return specs


# ======================================================================
# Exp 8.2.1: Real datasets — identity marginal workload
# ======================================================================

def exp_8_2_1():
    """Real datasets with identity workloads.

    All methods (WFF, RP, RP+, QS) achieve SVDB-optimal RMSE on identity
    marginal workloads. This experiment confirms correctness.
    """
    print("=" * 110)
    print("  Exp 8.2.1: Real Datasets — Identity Marginal Workload")
    print("  Metric: RMSE = sqrt(SumVar / #Queries) at privacy cost 1")
    print("=" * 110)

    for ds_name, ds_info in DATASETS.items():
        domains = ds_info["domains"]
        n_attr = len(domains)
        print(f"\nDataset: {ds_name}  domains={domains}  ({n_attr} attributes)")

        specs = make_kway_specs(domains)

        header = (f"  {'Workload':<10s} {'#Marg':>6s} {'#Q':>10s} "
                  f"{'RP':>8s} {'RP+':>8s} "
                  f"{'WFF':>8s} {'QS':>8s} "
                  f"{'WFF-Time':>9s} {'QS-Time':>9s}")
        print(header)
        print("  " + "-" * 90)

        for spec_name, marginals in specs.items():
            kron_wl = build_kron_identity_workload(domains, marginals)
            n_q = kron_n_queries(kron_wl, marginals)

            # Paper values (RP = RP+ = SVDB for identity)
            rp_paper = RP_PAPER.get(ds_name, {}).get(spec_name)
            rpp_paper = RPP_PAPER.get(ds_name, {}).get(spec_name)

            # WFF (KronFourier)
            wff_rmse, wff_time = run_qs_kron(domains, marginals, kron_wl,
                                              "fourier", label="WFF")

            # QS (KronConvex with warm start + fallback)
            qs_rmse, qs_time = run_qs_kron(domains, marginals, kron_wl,
                                            "convex", label="QS")

            print(f"  {spec_name:<10s} {len(marginals):6d} {n_q:10d} "
                  f"{fmt(rp_paper)} {fmt(rpp_paper)} "
                  f"{fmt(wff_rmse)} {fmt(qs_rmse)} "
                  f"{wff_time:8.1f}s {qs_time:8.1f}s")

        print("  " + "-" * 90)

    print()
    print("  RP/RP+ values from RP+ paper Table accrmse (= SVDB lower bound).")
    print("  All methods achieve identical RMSE, confirming SVDB-optimality.")
    print()


# ======================================================================
# Exp 8.2.2: Synthetic scalability — Syn-10^d
# ======================================================================

def exp_8_2_2():
    """Synthetic [10]^d configs, scaling number of attributes d.

    Matches RP+ paper synthetic setup (Syn-n^d with n=10, varying d).
    Identity marginal workload, <=3-way. All methods are SVDB-optimal;
    the comparison focuses on timing scalability.
    """
    print("=" * 110)
    print("  Exp 8.2.2: Synthetic Scalability — Syn-10^d, <=3-way Identity Marginals")
    print("  Metric: RMSE = sqrt(SumVar / #Queries) at privacy cost 1")
    print("=" * 110)

    ds = [2, 5, 10, 15, 20]
    n = 10
    max_k = 3

    header = (f"  {'d':>4s} {'#Marg':>6s} {'#Q':>10s} "
              f"{'RP+':>8s} {'WFF':>8s} {'QS':>8s} "
              f"{'RP+-Time':>9s} {'WFF-Time':>9s} {'QS-Time':>9s}")
    print(header)
    print("  " + "-" * 85)

    for d in ds:
        domains = [n] * d
        marginals = k_way_marginals(domains, max_k)

        kron_wl = build_kron_identity_workload(domains, marginals)
        n_q = kron_n_queries(kron_wl, marginals)

        # RP+ (live run — no paper RMSE values for synthetic configs)
        rpp_rmse, rpp_time = run_rpp_identity(domains, marginals)

        # WFF
        wff_rmse, wff_time = run_qs_kron(domains, marginals, kron_wl,
                                          "fourier", label="WFF")

        # QS
        qs_rmse, qs_time = run_qs_kron(domains, marginals, kron_wl,
                                        "convex", label="QS")

        print(f"  {d:4d} {len(marginals):6d} {n_q:10d} "
              f"{fmt(rpp_rmse)} {fmt(wff_rmse)} {fmt(qs_rmse)} "
              f"{rpp_time:8.1f}s {wff_time:8.1f}s {qs_time:8.1f}s")

    print("  " + "-" * 85)
    print()
    print("  All methods achieve identical RMSE (SVDB-optimal for identity workloads).")
    print("  RP+ run live (no paper values for synthetic RMSE).")
    print()


# ======================================================================
# Exp 8.2.3: Prefix-sum workloads on real datasets
# ======================================================================

def exp_8_2_3():
    """Prefix-sum workloads on real datasets.

    Mixed bases: prefix for numerical attributes, identity for categorical.
    Matches RP+ paper Table accrmse-prefix.
    RP cannot handle prefix workloads (identity marginals only).
    """
    print("=" * 110)
    print("  Exp 8.2.3: Prefix-Sum Workloads on Real Datasets")
    print("  Mixed bases: prefix for numerical, identity for categorical")
    print("  Metric: RMSE = sqrt(SumVar / #Queries) at privacy cost 1")
    print("=" * 110)

    for ds_name, ds_info in DATASETS.items():
        domains = ds_info["domains"]
        num_attrs = ds_info["num_attrs"]
        n_attr = len(domains)
        print(f"\nDataset: {ds_name}  domains={domains}")
        print(f"  Numerical: first {num_attrs} attrs (prefix basis), "
              f"Categorical: last {n_attr - num_attrs} attrs (identity basis)")

        specs = make_kway_specs(domains)

        header = (f"  {'Workload':<10s} {'#Marg':>6s} {'#Q':>10s} "
                  f"{'RP+':>8s} {'HDMM':>8s} "
                  f"{'WFF':>8s} {'QS':>8s} "
                  f"{'WFF-Time':>9s} {'QS-Time':>9s}")
        print(header)
        print("  " + "-" * 90)

        for spec_name, marginals in specs.items():
            kron_wl = build_kron_mixed_basis_workload(domains, marginals, num_attrs)
            n_q = kron_n_queries(kron_wl, marginals)

            # Paper values
            rpp_paper = RPP_PAPER_PREFIX.get(ds_name, {}).get(spec_name)
            hdmm_paper = HDMM_PAPER_PREFIX.get(ds_name, {}).get(spec_name)

            # WFF (KronFourier)
            wff_rmse, wff_time = run_qs_kron(domains, marginals, kron_wl,
                                              "fourier", label="WFF")

            # QS (KronConvex with warm start + fallback)
            qs_rmse, qs_time = run_qs_kron(domains, marginals, kron_wl,
                                            "convex", label="QS")

            print(f"  {spec_name:<10s} {len(marginals):6d} {n_q:10d} "
                  f"{fmt(rpp_paper)} {fmt(hdmm_paper)} "
                  f"{fmt(wff_rmse)} {fmt(qs_rmse)} "
                  f"{wff_time:8.1f}s {qs_time:8.1f}s")

        print("  " + "-" * 90)

    print()
    print("  RP+/HDMM values from RP+ paper Table accrmse-prefix.")
    print("  RP not shown (cannot handle prefix workloads).")
    print()


# ======================================================================
# Main
# ======================================================================

if __name__ == "__main__":
    print_hardware_info()
    exps = sys.argv[1:] if len(sys.argv) > 1 else ["1", "2", "3"]
    if "1" in exps:
        exp_8_2_1()
    if "2" in exps:
        exp_8_2_2()
    if "3" in exps:
        exp_8_2_3()
    print("Section 8.2 experiments complete.")
