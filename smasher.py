"""
Main QuerySmasher class.

Orchestrates the full pipeline:
1. Query decomposition
2. Partition workload
3. Optimize partitions independently
4. Assemble global mechanism
5. Measure and reconstruct
6. Answer queries
"""

from typing import List, Dict, Tuple, Optional, Set, Any
from itertools import combinations, chain
import numpy as np
from functools import reduce as _reduce

from query import Query, KronQuery
from solver import LowDimSolver, convexSolver, FourierSolver, KronConvexSolver, KronFourierSolver


def _compute_pbasis_res_mat(d: int) -> np.ndarray:
    """Compute the P-basis residual matrix for domain size d.

    Uses the McKennaConvex-style optimization on the prefix workload to find
    a residual matrix with low privacy cost (pcost < 1) and identity noise
    covariance.  This can provide a better mechanism than the Fourier-warm-
    started L-BFGS-B convex solver for mixed workloads.

    Returns:
        res_mat: (d-1, d) residual matrix with identity noise covariance.
    """
    from scipy import optimize
    from scipy.linalg.lapack import dpotrf, dpotri

    # Prefix workload: lower triangular matrix of 1s
    W = np.tril(np.ones((d, d)))

    # Center workload and prepend sum query
    proj = np.eye(d) - np.ones((d, 1)) @ np.ones((1, d)) / d
    workprime = np.vstack([np.ones((1, d)), W @ proj])
    V = workprime.T @ workprime

    # McKennaConvex: minimize trace(X^{-1} V) over X = I + off-diag, PSD
    mask = np.tri(d, dtype=bool, k=-1)

    def loss_and_grad(params, V=V, mask=mask, d=d):
        X = np.zeros((d, d))
        X[mask] = params
        X += X.T
        np.fill_diagonal(X, 1.0)
        zz, info0 = dpotrf(X, False, False)
        iX, info1 = dpotri(zz)
        iX = np.triu(iX) + np.triu(iX, k=1).T
        if info0 != 0 or info1 != 0:
            return 1e10, np.zeros_like(params)
        loss = np.sum(iX * V)
        G = -iX @ V @ iX
        return loss, G[mask] + G.T[mask]

    # Warm start from matrix square root of V
    eig, P = np.linalg.eigh(V)
    eig = np.real(eig)
    eig[eig < 1e-10] = 0.0
    X0 = P @ np.diag(np.sqrt(eig)) @ P.T
    X0 /= np.diag(X0).max()

    res = optimize.minimize(loss_and_grad, X0[mask], jac=True,
                            method='L-BFGS-B', options={'maxcor': 1})

    # Extract strategy matrix
    Xfinal = np.zeros((d, d))
    Xfinal[mask] = res.x
    Xfinal += Xfinal.T
    np.fill_diagonal(Xfinal, 1.0)
    Si = np.linalg.cholesky(Xfinal).T

    # Compute residual matrix (Algorithm 4 from RP+)
    Bs = np.ones((1, d))
    P1 = Si - Si @ Bs.T @ Bs / d
    gram = P1.T @ P1 + np.eye(d) * 1e-10
    L = np.linalg.cholesky(gram)
    col_norms = np.linalg.norm(L, axis=0)
    keep = np.sort(np.argsort(col_norms)[-(d - 1):])
    return L[:, keep].T  # (d-1, d)


# Cache for P-basis residual matrices (keyed by domain size)
_pbasis_cache: Dict[int, np.ndarray] = {}

# Optional JAX solver (GPU-accelerated)
try:
    import sys as _sys
    import os as _os
    _jax_path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'jax_qs')
    if _jax_path not in _sys.path:
        _sys.path.insert(0, _jax_path)
    from jax_solver import JaxConvexSolver
    JAX_AVAILABLE = True
except ImportError:
    JAX_AVAILABLE = False

# Helper to get power set
def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

class Smasher:
    """
    Main QuerySmasher orchestrator.
    """
    
    def __init__(self, domains: List[int], privacy_cost: float = 1.0,
                 default_solver: str = 'fourier', noise: bool = True,
                 verbose: bool = True, convex_optimizer: str = 'lbfgsb_fourier',
                 convex_kwargs: dict = None, hybrid_kron: bool = False,
                 hybrid_kron_threshold: int = 2000, use_jax: bool = False):
        """
        Initialize QuerySmasher.

        Args:
            domains: List of domain sizes for all attributes.
            privacy_cost: Target privacy cost (beta).
            default_solver: Default solver type ('fourier' or 'convex').
            noise: If True (default), add Gaussian noise during measurement.
                   If False, measurements are noiseless (useful for testing/debugging).
            verbose: If True, print progress.
            convex_optimizer: Optimizer for convexSolver ('lbfgsb', 'lbfgsb_fourier', 'scs').
            convex_kwargs: Extra keyword args for convexSolver (mu_schedule, ftol, maxcor).
            hybrid_kron: If True, use dense convex for small multi-attribute KronQuery
                         partitions (r <= threshold) instead of KronConvex. Dense convex
                         is optimal (no Kronecker restriction).
            hybrid_kron_threshold: Max r=prod(d_i-1) for using dense convex in hybrid
                                   mode. Default 2000.
        """
        self.domains = domains
        self.privacy_cost = privacy_cost
        self.default_solver = default_solver
        self.noise = noise
        self.verbose = verbose
        self.convex_optimizer = convex_optimizer
        self.convex_kwargs = convex_kwargs or {}
        self.hybrid_kron = hybrid_kron
        self.hybrid_kron_threshold = hybrid_kron_threshold
        self.use_jax = use_jax and JAX_AVAILABLE

        # Workload management
        self.queries: List[Query] = []
        self.partitions: Dict[Tuple[int, ...], List[Query]] = {}
        
        # Solver management
        self.solvers: Dict[Tuple[int, ...], LowDimSolver] = {}
        self.solver_types: Dict[Tuple[int, ...], str] = {}
        self._kron_factor_cache: Dict = {}  # Shared cache for KronConvexSolver per-factor results
        self._dense_partition_cache: Dict = {}  # gram_hash -> (B, V_pinv, L, solver_type)
        
        # Measurement results
        self.measurements: Dict[Tuple[int, ...], Any] = {}
        self.reconstructed_marginals: Dict[Tuple[int, ...], np.ndarray] = {}
        self.data_vector: Optional[np.ndarray] = None
        
        # Pipeline state
        self.is_decomposed = False
        self.is_optimized = False
        self.is_measured = False
        self.is_reconstructed = False
    
    def add_queries_to_workload(self, queries: List[Query]):
        """
        Add Query objects to the workload.
        """
        self.queries.extend(queries)
        # Invalidate previous work
        self.is_decomposed = False
        self.is_optimized = False
        self.partitions = {}
        if self.verbose:
            print(f"Added {len(queries)} queries. Total: {len(self.queries)}")
    
    def set_solver_for_partition(self, attributes: Tuple[int, ...], solver_type: str):
        """
        Override default solver for a specific partition.
        """
        self.solver_types[attributes] = solver_type
    
    def __get_solver_type_for_partition(self, subset: Tuple[int, ...]) -> str:
        """
        Determine which solver type will be used for a partition.

        Args:
            subset: The attribute subset (partition key)

        Returns:
            Solver type string ('convex' or 'fourier')
        """
        return self.solver_types.get(subset, self.default_solver)

    def __decompose_workload(self):
        """
        STEP 1: Decompose workload.

        Note: For convex solver, we use a different decomposition (Section 7.1).
        Instead of using the centering matrix (I - (1/d)*J) for attributes in the
        subset, we use the identity matrix I. The convex solver will then project
        onto the residual subspace after optimization.
        """
        if self.verbose:
            print("Decomposing workload...")

        self.partitions = {}

        for q in self.queries:
            A = q.attributes_used()
            # Decompose into all subsets A' subseteq A
            # Sort A to ensure consistent tuple keys
            # Although attributes should be handled consistently.
            # Assuming A is a tuple of sorted indices.

            for A_prime in powerset(A):
                # Ensure A_prime is a tuple
                A_prime = tuple(sorted(A_prime))

                # Determine if convex solver will be used for this partition
                # Section 7.1: use identity matrix instead of centering matrix
                solver_type = self.__get_solver_type_for_partition(A_prime)
                use_convex_decomposition = (solver_type == 'convex')

                # Decompose query with appropriate method
                sub_q = q.decompose(A_prime, for_convex=use_convex_decomposition)

                # Add to partition
                if A_prime not in self.partitions:
                    self.partitions[A_prime] = []
                self.partitions[A_prime].append(sub_q)
        
        self.is_decomposed = True
        if self.verbose:
            print(f"Decomposition complete. Created {len(self.partitions)} partitions.")
    
    def __all_kron_queries(self, subset: Tuple[int, ...]) -> bool:
        """Check if all queries in a partition are KronQuery."""
        queries = self.partitions.get(subset, [])
        return len(queries) > 0 and all(isinstance(q, KronQuery) for q in queries)

    def __solve_partition(self, subset: Tuple[int, ...],
                         solver_type: Optional[str] = None) -> LowDimSolver:
        """
        STEP 2: Optimize a single partition independently.

        When solver_type is 'convex', also solves with Fourier and keeps
        whichever achieves lower L_A' (weighted sum of variances). This
        prevents the convex solver's log-barrier interior bias from
        degrading near-optimal Fourier solutions.

        When all queries in the partition are KronQuery and the partition has
        multiple attributes, uses KronConvexSolver/KronFourierSolver which
        solve per-factor subproblems independently. Falls back to regular
        solvers for single-attribute partitions or mixed query types.
        """
        if solver_type is None:
            solver_type = self.solver_types.get(subset, self.default_solver)

        # Check if we can use Kron-factored solvers
        all_kron = (len(subset) > 1 and self.__all_kron_queries(subset))
        r = int(np.prod([self.domains[i] - 1 for i in subset])) if subset else 1

        # Hybrid mode: use dense convex for small KronQuery partitions
        if all_kron and self.hybrid_kron and r <= self.hybrid_kron_threshold:
            use_kron = False
            kron_tag = f" [hybrid-dense r={r}]"
        elif all_kron:
            use_kron = True
            kron_tag = " [kron]"
        else:
            use_kron = False
            kron_tag = ""

        if self.verbose:
            print(f"  Solving partition {subset} with {solver_type}{kron_tag}...")

        if solver_type == 'convex':
            # Adaptive convex kwargs based on partition size r
            if not self.convex_kwargs:
                # For Kron solvers, per-factor r_i = d_i - 1 is always small,
                # so always use tight settings regardless of full product r.
                max_factor_r = max((self.domains[i] - 1 for i in subset), default=0) if use_kron else r
                if use_kron and max_factor_r <= 500:
                    # Kron per-factor subproblems: profiled-tight settings
                    adaptive_kwargs = {"mu_schedule": [0.1, 0.01, 0.001, 0.0001],
                                       "ftol": 1e-10, "maxcor": 50}
                elif max_factor_r <= 500:
                    # Dense small partitions: tight optimization is cheap
                    adaptive_kwargs = {"mu_schedule": [0.1, 0.01, 0.001],
                                       "ftol": 1e-6, "maxcor": 10}
                else:
                    # Dense large partitions (r > 500): use aggressive settings
                    # to avoid extremely long solves while still getting improvement
                    adaptive_kwargs = {"mu_schedule": [0.1, 0.01],
                                       "ftol": 1e-4, "maxcor": 10}
            else:
                adaptive_kwargs = self.convex_kwargs

            # --- Dense partition cache: skip redundant solves ---
            cache_key = None
            if not use_kron:
                # Build query matrix and gram to check cache
                _tmp_solver = convexSolver(self.domains, subset,
                                           optimizer=self.convex_optimizer, **adaptive_kwargs)
                for q in self.partitions[subset]:
                    _tmp_solver.add_query(q)
                W, weights = _tmp_solver._build_query_matrix()
                if W is not None:
                    U = _tmp_solver._get_residual_basis()
                    W_proj = (W * np.sqrt(weights[:, None])) @ U
                    A = W_proj.T @ W_proj
                    # Include partition shape in key to avoid cross-partition collisions
                    shape = tuple(self.domains[i] for i in subset)
                    cache_key = (shape, np.round(A, decimals=8).tobytes())

                if cache_key is not None and cache_key in self._dense_partition_cache:
                    cached = self._dense_partition_cache[cache_key]
                    winner_type = cached['solver_type']
                    if self.verbose:
                        print(f"    Cache hit ({winner_type}) for {subset}")
                    if winner_type == 'convex':
                        solver = convexSolver(self.domains, subset,
                                              optimizer=self.convex_optimizer, **adaptive_kwargs)
                        for q in self.partitions[subset]:
                            solver.add_query(q)
                        solver._restore_from_cache(cached['B'], cached['V_pinv'])
                    else:
                        solver = FourierSolver(self.domains, subset)
                        for q in self.partitions[subset]:
                            solver.add_query(q)
                        solver._restore_from_cache(cached['theta'], cached['c'],
                                                   cached['theta_array'])
                    return solver

            if use_kron:
                # Kron-factored convex solver (per-factor subproblems)
                convex_solver = KronConvexSolver(
                    self.domains, subset,
                    optimizer=self.convex_optimizer,
                    factor_cache=self._kron_factor_cache,
                    **adaptive_kwargs)
            elif self.use_jax:
                # JAX/GPU solver for dense partitions
                jax_kwargs = {k: v for k, v in adaptive_kwargs.items()
                              if k in ('mu_schedule', 'ftol')}
                convex_solver = JaxConvexSolver(
                    self.domains, list(subset), **jax_kwargs)
                for q in self.partitions[subset]:
                    convex_solver.add_query(q)
            else:
                convex_solver = _tmp_solver  # reuse already-constructed solver
            if use_kron:
                for q in self.partitions[subset]:
                    convex_solver.add_query(q)
            convex_solver.optimize(target_privacy_cost=1.0)

            # Also solve with Fourier for comparison
            if use_kron:
                fourier_solver = KronFourierSolver(self.domains, subset)
            else:
                fourier_solver = FourierSolver(self.domains, subset)
            for q in self.partitions[subset]:
                fourier_solver.add_query(q)
            fourier_solver.optimize(target_privacy_cost=1.0)

            # Compare L_A' and keep the better one
            convex_L = convex_solver.weighted_total_variance()
            fourier_L = fourier_solver.weighted_total_variance()

            # --- P-basis candidate for dense (non-Kron) partitions ---
            # The P-basis (prefix-optimized) mechanism can beat both convex and
            # Fourier when the subquery workload is non-circulant (e.g. mixed
            # range+affine+prefix).  The convex solver's L-BFGS-B, warm-started
            # from Fourier, can get stuck in a local minimum far from this
            # solution.  We evaluate the P-basis mechanism as a third candidate.
            pbasis_L = float('inf')
            pbasis_solver = None
            if not use_kron and subset:
                try:
                    pbasis_solver, pbasis_L = self.__try_pbasis_mechanism(
                        subset, convex_solver, adaptive_kwargs)
                except Exception:
                    pass  # P-basis unavailable or failed; ignore

            best_L = min(convex_L, fourier_L, pbasis_L)

            if best_L == fourier_L and fourier_L <= convex_L:
                if self.verbose:
                    print(f"    Fourier wins on {subset}: {fourier_L:.2f} < {convex_L:.2f}")
                if cache_key is not None:
                    self._dense_partition_cache[cache_key] = {
                        'solver_type': 'fourier',
                        'theta': fourier_solver.theta,
                        'c': fourier_solver.c,
                        'theta_array': getattr(fourier_solver, '_theta_array', None),
                    }
                return fourier_solver
            elif pbasis_solver is not None and pbasis_L < convex_L:
                if self.verbose:
                    print(f"    P-basis wins on {subset}: {pbasis_L:.2f} < convex {convex_L:.2f}")
                if cache_key is not None:
                    self._dense_partition_cache[cache_key] = {
                        'solver_type': 'convex',
                        'B': pbasis_solver.B,
                        'V_pinv': pbasis_solver._V_pinv,
                    }
                return pbasis_solver
            # Convex wins (or ties)
            if cache_key is not None:
                self._dense_partition_cache[cache_key] = {
                    'solver_type': 'convex',
                    'B': convex_solver.B,
                    'V_pinv': convex_solver._V_pinv,
                }
            return convex_solver

        elif solver_type == 'fourier':
            if use_kron:
                solver = KronFourierSolver(self.domains, subset)
            else:
                # Dense Fourier cache: check before solving
                solver = FourierSolver(self.domains, subset)
                for q in self.partitions[subset]:
                    solver.add_query(q)
                # Build query matrix for cache key
                rows, wts = [], []
                for q in solver.queries:
                    qv = q.query_vector()
                    if qv.ndim > 1:
                        for i in range(qv.shape[0]):
                            rows.append(qv[i]); wts.append(q.weight)
                    else:
                        rows.append(qv); wts.append(q.weight)
                if rows and len(subset) > 0:
                    W = np.vstack(rows)
                    weights = np.array(wts)
                    Ww = W * np.sqrt(weights[:, None])
                    gram = Ww.T @ Ww
                    shape = tuple(self.domains[i] for i in subset)
                    fkey = ('fourier', shape, hash(np.round(gram, 8).tobytes()))
                    if fkey in self._dense_partition_cache:
                        cached = self._dense_partition_cache[fkey]
                        if self.verbose:
                            print(f"    Cache hit (fourier) for {subset}")
                        solver._restore_from_cache(cached['theta'], cached['c'],
                                                   cached['theta_array'])
                        return solver
                    solver.optimize(target_privacy_cost=1.0)
                    self._dense_partition_cache[fkey] = {
                        'theta': solver.theta,
                        'c': solver.c,
                        'theta_array': getattr(solver, '_theta_array', None),
                    }
                    return solver
                # Trivial partition (empty subset) — just optimize directly
                solver.optimize(target_privacy_cost=1.0)
                return solver
        else:
            raise ValueError(f"Unknown solver type: {solver_type}")

        # Add queries to solver (kron fourier path)
        for q in self.partitions[subset]:
            solver.add_query(q)

        # Optimize
        solver.optimize(target_privacy_cost=1.0)

        return solver
    
    def __try_pbasis_mechanism(self, subset, template_solver, adaptive_kwargs):
        """Evaluate the P-basis (prefix-optimized) mechanism for a dense partition.

        Computes the Kronecker product of per-attribute P-basis residual matrices,
        scales to pcost=1, and evaluates L on the partition's queries.

        Returns:
            (solver, L): A convexSolver with the P-basis mechanism restored, and
                         its weighted total variance.  Returns (None, inf) on failure.
        """
        global _pbasis_cache

        # Compute per-attribute P-basis residual matrices (cached by domain size)
        res_mats = []
        for attr_idx in subset:
            d = self.domains[attr_idx]
            if d not in _pbasis_cache:
                _pbasis_cache[d] = _compute_pbasis_res_mat(d)
            res_mats.append(_pbasis_cache[d])

        # B_partition = kron(res_mat_1, ..., res_mat_k), shape (prod(d_i-1), prod(d_i))
        B = _reduce(np.kron, res_mats)

        # V = B^T B; scale to pcost=1 (max diagonal = 1)
        V = B.T @ B
        pcost = np.max(np.diag(V))
        if pcost < 1e-12:
            return None, float('inf')
        B_scaled = B / np.sqrt(pcost)

        # Project into residual subspace and compute V_pinv
        U = template_solver._get_residual_basis()
        S_proj = U.T @ (B_scaled.T @ B_scaled) @ U
        S_proj_inv = np.linalg.inv(S_proj + 1e-12 * np.eye(S_proj.shape[0]))
        V_pinv = U @ S_proj_inv @ U.T

        # Create solver with P-basis mechanism
        pbasis_solver = convexSolver(self.domains, subset,
                                     optimizer=self.convex_optimizer, **adaptive_kwargs)
        for q in self.partitions[subset]:
            pbasis_solver.add_query(q)
        pbasis_solver._restore_from_cache(B_scaled, V_pinv)

        return pbasis_solver, pbasis_solver.weighted_total_variance()

    def __solve_all_partitions(self, use_parallel: bool = False):
        """
        STEP 2: Solve all partitions independently.
        """
        if not self.partitions:
            if self.verbose: print("No partitions to solve.")
            return

        # Parallel implementation omitted for simplicity, iterating sequentially
        for subset in self.partitions:
             # Just use default logic for solver choice if not set
            self.solvers[subset] = self.__solve_partition(subset)
            
    def __assemble(self):
        """
        STEP 3: Assemble global mechanism via noise rescaling.
        """
        if self.verbose: print("Assembling global mechanism...")
        
        # Calculate L_A' for each partition
        # L_A' is the weighted sum of variances (objective function value)
        # for privacy_cost = 1.0
        L_values = {}
        
        for subset, solver in self.solvers.items():
            L_values[subset] = solver.weighted_total_variance()
            
        # Calculate Gamma = sum(sqrt(L_A')) / pcost
        sqrt_L_sum = sum(np.sqrt(val) for val in L_values.values())
        
        if self.privacy_cost <= 0:
            raise ValueError("Privacy cost must be positive")
            
        gamma = sqrt_L_sum / self.privacy_cost
        
        # Rescale each solver
        # sigma^2_A' = gamma / sqrt(L_A')
        # Check for L_A' = 0 case
        
        for subset, solver in self.solvers.items():
            L = L_values[subset]
            if L > 1e-9:
                sigma_sq = gamma / np.sqrt(L)
            else:
                sigma_sq = 0.0 # If variance is 0 (trivial queries), no noise needed? 
                               # Or maybe infinite noise if L is 0? 
                               # If L is 0, it contributes 0 to sqrt_L_sum.
                               # It implies 0 error at cost 1. So we can put 0 cost?
                               # Actually if L=0, we don't need to allocate budget.
            
            # Rescale noise: The solver has noise Sigma for cost 1.
            # We want noise for cost (1/sigma^2)?
            # Wait, formula: M* = B x + N(0, sigma^2 * Sigma)
            # So we multiply covariance by sigma^2.
            solver.rescale_noise(sigma_sq)
            
    def __extract_marginal(self, data_vector: np.ndarray, 
                          subset: Tuple[int, ...]) -> np.ndarray:
        """
        Extract marginal x_A from full data vector.
        """
        # Reshape data_vector to tensor
        full_shape = tuple(self.domains)
        data_tensor = data_vector.reshape(full_shape)
        
        # Determine axes to sum over (those NOT in subset)
        # subset contains indices of axes to KEEP
        axes_to_sum = tuple(i for i in range(len(self.domains)) if i not in subset)
        
        if not axes_to_sum:
            # Keep all axes
            marginal_tensor = data_tensor
        else:
            marginal_tensor = np.sum(data_tensor, axis=axes_to_sum)
            
        # Flatten result. But need to ensure correct order of axes?
        # If subset is (2, 0), the marginal should probably be ordered (0, 2) or (2, 0)?
        # Our Query decomposition assumes a specific order logic.
        # Usually we assume standard sorted attribute order.
        # But if we summed over axes, the remaining axes remain in their relative order.
        # E.g. (A, B, C), sum over B -> (A, C). This is sorted if original was sorted.
        
        return marginal_tensor.flatten()

    def optimize(self, use_parallel: bool = False):
        """
        Run optimization pipeline.
        """
        import time as _time

        if not self.queries:
            print("Warning: Workload is empty.")
            return

        # 1. Decompose
        t0 = _time.time()
        self.__decompose_workload()
        t_decompose = _time.time() - t0

        # 2. Solve Partitions
        t0 = _time.time()
        self.__solve_all_partitions(use_parallel)
        t_solve = _time.time() - t0

        # 3. Assemble
        t0 = _time.time()
        self.__assemble()
        t_assemble = _time.time() - t0

        self.is_optimized = True
        self.phase_times = {
            "decompose": t_decompose,
            "solve": t_solve,
            "assemble": t_assemble,
        }
        if self.verbose: print("Optimization complete.")

    def measure(self, data_vector: np.ndarray):
        """
        Apply mechanisms and reconstruct marginals.
        """
        if not self.is_optimized:
            raise RuntimeError("Must call optimize() before measure()")
            
        if self.verbose: print("Measuring data...")
        self.data_vector = data_vector
        self.measurements = {}
        self.reconstructed_marginals = {}
        
        for subset, solver in self.solvers.items():
            # Extract marginal
            x_A = self.__extract_marginal(data_vector, subset)

            # Apply mechanism (with or without noise based on self.noise)
            noisy_meas = solver.measure(x_A, add_noise=self.noise)
            self.measurements[subset] = noisy_meas
            
            # Reconstruct marginal
            recon_x_A = solver.reconstruct_marginal(noisy_meas)
            self.reconstructed_marginals[subset] = recon_x_A
            
        self.is_measured = True
        self.is_reconstructed = True # We do reconstruction immediately here
        if self.verbose: print("Measurement complete.")
        
    def answer_query(self, query: Query) -> float:
        """
        Answer a query from reconstructed marginals.

        Note: Uses the same decomposition method (convex vs standard) that was
        used during optimization for each partition.

        For multi-row queries (MatrixQuery with multiple rows, or KronQuery),
        returns a numpy array of answers, one per query row.
        """
        if not self.is_reconstructed:
            raise RuntimeError("Must call measure() before answer_query()")

        # For KronQuery with multiple factors, decompose() may change the
        # row count for partial subsets (collapsed factors reduce rows).
        # Handle by answering each original row independently.
        q_vec = query.query_vector()
        if isinstance(query, KronQuery) and q_vec.ndim > 1 and q_vec.shape[0] > 1:
            from query import MatrixQuery as _MQ
            results = np.zeros(q_vec.shape[0])
            for i in range(q_vec.shape[0]):
                row_q = _MQ(query.domains, query.attributes_used(),
                            q_vec[i:i+1], weight=query.weight)
                ans = self.answer_query(row_q)
                results[i] = float(np.asarray(ans).flat[0])
            return results

        A = query.attributes_used()

        # We need to sum answers from all subqueries
        total_answer = 0.0

        # Decompose input query same way we did in optimize
        # Note: the input query might not be in the original workload!
        # But we must decompose it to use our reconstructed marginals.

        for A_prime in powerset(A):
            A_prime = tuple(sorted(A_prime))

            if A_prime in self.reconstructed_marginals:
                # Use same decomposition method as during optimization
                solver_type = self.__get_solver_type_for_partition(A_prime)
                use_convex_decomposition = (solver_type == 'convex')

                # Get subquery with appropriate decomposition
                sub_q = query.decompose(A_prime, for_convex=use_convex_decomposition)

                # Get reconstructed marginal
                x_hat = self.reconstructed_marginals[A_prime]

                # Answer subquery: q_sub . x_hat
                q_vec = sub_q.query_vector()
                ans = q_vec @ x_hat

                if isinstance(ans, np.ndarray):
                    total_answer += ans
                else:
                    total_answer += ans

        return total_answer

    def workload_variance(self) -> float:
        """Compute weighted sum of variances for entire workload."""
        total_var = 0.0
        for subset, solver in self.solvers.items():
            # Variance is additive across partitions (independence)
            total_var += solver.weighted_total_variance()
        return total_var