# QuerySmasher

A scalable, differentially private matrix mechanism for optimally answering complex workloads of linear queries.

QuerySmasher decomposes a workload of linear queries into independent sub-problems via Kronecker structure, solves each optimally, and assembles the results with a global noise budget using Cauchy-Schwarz optimal allocation.

## Setup

**Requirements:** Python 3.11+

```bash
pip install numpy scipy cvxpy torch
```

Or install from the requirements file:

```bash
pip install -r requirements.txt
```

## Quick Start

```python
import numpy as np
from query import MatrixQuery, KronQuery
from smasher import Smasher

# Define a dataset with 3 attributes, each of domain size 5
domains = [5, 5, 5]

# Create QuerySmasher
qs = Smasher(domains, privacy_cost=1.0, default_solver='convex', verbose=False)

# Add prefix queries on attribute 0
P = np.tril(np.ones((5, 5)))  # prefix query matrix
qs.add_queries_to_workload([MatrixQuery(domains, (0,), P)])

# Add prefix queries on attributes (0, 1) using Kronecker structure
qs.add_queries_to_workload([KronQuery(domains, (0, 1), [P, P])])

# Optimize the mechanism
qs.optimize()

# Measure a data vector (with calibrated Gaussian noise)
data = np.random.randint(0, 10, size=np.prod(domains)).astype(float)
qs.measure(data)

# Answer a query
answer = qs.answer_query(MatrixQuery(domains, (0,), P))
print("Noisy prefix sums on attribute 0:", answer)
```

## Running Tests

```bash
python -m unittest discover tests/ -v
```

74 tests covering query decomposition, solver correctness, reconstruction, privacy cost, and end-to-end pipelines.

## Reproducing Paper Results

All experiment scripts are in `exp/`. Run each script from the `exp/` directory:

```bash
cd exp
```

**Note:** Experiments on large configurations (e.g., `n=40, d=40`) require significant memory (up to 230 GB) and may take hours. Smaller configurations (e.g., `n=10, d=10`) run in seconds.

### Table 1 -- Prefix Workloads on Real Data

RMSE for hybrid marginal/prefix workloads on Adult, CPS, and Loans datasets at 1-way, 2-way, 3-way, and up-to-3-way levels. Compares RP+, HDMM, WFF, and QS.

```bash
python -u paper_section8_2.py
```

### Table 2 -- Affine and Abs-Diff Workloads on Real Data

RMSE for 2-way affine and abs-diff queries on CPS, Adult, and Loans. These query types are only supported by WFF and QS.

```bash
python -u paper_real_affine_abs.py
```

### Table 3 -- 2-Way Workloads on Synthetic Data

RMSE of 1-way + 2-way workloads on `[n]^40` (40 attributes, varying domain size `n = 10..50`) across 7 query types: marginal, circ_range, range, prefix, affine, abs_diff, random.

```bash
python -u paper_2way_nd.py            # WFF and QS columns
python -u paper_baseline_fill.py      # RP+ and HDMM baselines
python -u paper_2way_nd_jax_fill2.py  # Fill remaining cells for n=40,50
```

### Table 4 -- Mixed Workload Accuracy

RMSE on mixed-query workloads combining 1-way range queries, 2-way affine queries, and 3-way prefix queries on `[n]^d` as both `n` and `d` vary.

```bash
python -u paper_mixed_nd.py
```

### Table 5 -- Prefix Workload Timing

Timing (decompose/solve/assemble phases) and peak memory for 1-way + 2-way prefix queries on `[n]^d`.

```bash
python -u paper_prefix_timing_fill.py
```

### Table 6 -- Mixed Workload Timing

Timing and peak memory for mixed-query workloads. Produced by the same script as Table 4:

```bash
python -u paper_mixed_nd.py
```

## Project Structure

```
querySmasher/
  query.py            Query classes: MatrixQuery (dense) and KronQuery (Kronecker-factored)
  solver.py           Low-dimensional solvers: convexSolver, FourierSolver,
                      KronConvexSolver, KronFourierSolver
  smasher.py          Main orchestrator (decompose -> solve -> assemble -> measure -> reconstruct)
  requirements.txt    Python dependencies
  tests/              74 unit tests
  exp/                Experiment scripts for reproducing paper tables
    exp_utils.py      Shared utilities: workload builders, RP+ integration, dataset definitions
  baselines/
    resplan/          ResidualPlanner+ (RP+) baseline implementation
```

## Baselines

- **RP+** (ResidualPlanner+): Included in `baselines/resplan/`. No additional setup required.
- **WFF** (Weighted Fourier Factorization): Implemented as `FourierSolver` within QuerySmasher itself.
- **HDMM** (High-Dimensional Matrix Mechanism): External dependency. To run HDMM comparisons, clone https://github.com/dpcomp-org/hdmm into `baselines/hdmm/`. HDMM is optional; experiments will report `NI` (not implemented) for HDMM columns if it is not installed.
