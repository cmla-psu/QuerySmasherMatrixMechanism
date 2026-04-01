"""
Low-dimensional solvers for QuerySmasher partitions.

Defines abstract LowDimSolver interface and concrete implementations:
- LowDimConvexSolver: Uses convex optimization
- FourierSolver: Uses Fourier-based mechanisms
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional
import numpy as np
import torch
from functools import reduce

from query import Query, KronQuery


class LowDimSolver(ABC):
    """
    Abstract base class for low-dimensional solvers (Section 7 of paper).

    Each solver optimizes a workload partition WKLoad^{⇒A'} independently
    at privacy cost = 1.0, then noise is rescaled in the assembly step.

    Paper notation:
        A' = subset of attributes for this partition
        WKLoad^{⇒A'} = workload partition containing decomposed subqueries
        M_A' = mechanism for this partition: M_A'(x̄_A') = B_A' x̄_A' + N(0, Σ_A')
        L_A' = weighted sum of variances for queries in this partition
        σ²_A' = noise scale factor (set by assembly step)

    Subclasses must implement:
        optimize(): Find optimal mechanism for the workload partition
        measure(x_marginal): Apply mechanism to marginal data
        reconstruct_marginal(omega): Reconstruct marginal from noisy measurements
        query_variance(query): Compute variance for a query
    """

    def __init__(self, domains: List[int], attributes: Tuple[int, ...]):
        """
        Args:
            domains: Full list of domain sizes [d_1, d_2, ..., d_k]
            attributes: Attributes A' = (i_1, ..., i_l) used by this partition
        """
        self.domains = domains
        self.attributes = attributes  # A' in paper notation
        self.queries: List[Query] = []  # WKLoad^{⇒A'}
        self.is_optimized = False

        # Set by optimize()
        self.Sigma: Optional[np.ndarray] = None  # Noise covariance Σ_A'
        self.privacy_cost_per_unit = 1.0  # Privacy cost before rescaling

        # Set by assembly step (rescale_noise)
        # This is σ²_A' = γ / √L_A' from Algorithm 1, Line 15
        self.noise_scale = 1.0
    
    def add_query(self, query: Query):
        """
        Add a decomposed subquery q^{⇒A'} to this partition's workload.

        Args:
            query: A Query object (typically from q.decompose(A'))
        """
        self.queries.append(query)
        self.is_optimized = False

    @abstractmethod
    def optimize(self, target_privacy_cost: float = 1.0):
        """
        Find optimal mechanism M_A' for this partition's workload (Lines 7-12).

        Solves: minimize L_A' = Σ_q weight(q) × Var(q)
                subject to privacy_cost ≤ target_privacy_cost
        """
        pass

    @abstractmethod
    def measure(self, x_marginal: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Apply mechanism to marginal data (Line 18).

        ω_A' = M*_A'(x̄_A')

        Args:
            x_marginal: Marginal data vector x̄_A'
            add_noise: If True, add Gaussian noise. If False, return noiseless measurement.

        Returns:
            omega: Noisy measurement ω_A' (or noiseless if add_noise=False)
        """
        pass

    @abstractmethod
    def reconstruct_marginal(self, omega: np.ndarray) -> np.ndarray:
        """
        Reconstruct marginal from noisy measurements.

        Uses Recon_A' function (Definition 4 in paper).

        Args:
            omega: Noisy measurement ω_A' from measure()

        Returns:
            x_hat: Reconstructed marginal x̂_A'
        """
        pass

    @abstractmethod
    def query_variance(self, query: Query) -> float:
        """
        Compute variance of a query answer under this mechanism.

        Uses Var_A' function (Definition 4 in paper).

        Args:
            query: Query q^{⇒A'} to compute variance for

        Returns:
            Variance of the reconstructed answer to query
        """
        pass

    def rescale_noise(self, sigma_sq: float):
        """
        Rescale noise covariance by σ²_A' (Assembly step, Lines 14-16).

        From Algorithm 1:
            σ²_A' = γ / √L_A'
            M*_A'(x̄_A') = B_A' x̄_A' + N(0, σ²_A' Σ_A')

        Args:
            sigma_sq: The σ²_A' value computed in assembly
        """
        self.noise_scale = sigma_sq
        if self.Sigma is not None:
            self.Sigma *= sigma_sq

    def get_privacy_cost(self) -> float:
        """Return the privacy cost of this mechanism."""
        return self.privacy_cost_per_unit

    def weighted_total_variance(self) -> float:
        """Compute L_A' = Σ_q weight(q) * Var(q) for all queries in this partition.

        Default implementation loops over queries. Subclasses may override
        with a batched implementation.
        """
        return sum(q.weight * self.query_variance(q) for q in self.queries)


class convexSolver(LowDimSolver):
    """
    Convex optimizer for low-dimensional partitions.

    Implements Section 7.1 of the QuerySmasher paper.

    Supports three optimizer backends:
        'lbfgsb' (default): L-BFGS-B in projected residual subspace with
            log-barrier for diag(U S U^T) <= target, Cholesky PSD enforcement.
        'lbfgsb_fourier': Same as lbfgsb but warm-started from Fourier solution
            so the initial S matches Fourier quality exactly.
        'scs': SDP via CVXPY/SCS in projected residual subspace.

    Paper notation mapping:
        W = query matrix (n_queries × marginal_size), each row is a query
        D = diagonal weight matrix, D[i,i] = weight of i-th query
        V = optimization variable (PSD), satisfies V = B^T B
        B = mechanism matrix, mechanism is M(x̄_A') = B x̄_A' + N(0, I)
        Σ (Sigma) = noise covariance matrix (identity in this formulation)
        P = projection matrix from Equation 3: ⊗_i (I_{d_i} - (1/d_i) J_{d_i})
        x̄_A' = marginal data vector
        ω (omega) = noisy measurement output from mechanism
    """

    OPTIMIZERS = ('lbfgsb', 'lbfgsb_fourier', 'scs',
                  'newton', 'newton_fourier',
                  'hybrid', 'hybrid_fourier',
                  'adaptive', 'adaptive_fourier')

    def __init__(self, domains: List[int], attributes: Tuple[int, ...],
                 optimizer: str = 'lbfgsb_fourier', *,
                 mu_schedule=None, ftol=1e-3, maxcor=10,
                 early_stop_rtol=None, nfev_budget=None,
                 adapt_mu_start=0.1, adapt_mu_factor=0.3,
                 adapt_mu_min=0.001, adapt_rtol=0.005,
                 adapt_max_rounds=10):
        super().__init__(domains, attributes)
        if optimizer not in self.OPTIMIZERS:
            raise ValueError(f"Unknown optimizer '{optimizer}', expected one of {self.OPTIMIZERS}")
        self.optimizer = optimizer
        self.mu_schedule = mu_schedule if mu_schedule is not None else [0.1]
        self.ftol = ftol
        self.maxcor = maxcor
        # Early stopping parameters
        self.early_stop_rtol = early_stop_rtol  # stop when relative improvement < rtol
        self.nfev_budget = nfev_budget           # max total function evaluations
        # Adaptive barrier parameters
        self._adapt_mu_start = adapt_mu_start
        self._adapt_mu_factor = adapt_mu_factor
        self._adapt_mu_min = adapt_mu_min
        self._adapt_rtol = adapt_rtol
        self._adapt_max_rounds = adapt_max_rounds
        self.B: Optional[np.ndarray] = None  # Mechanism matrix B
        self.W: Optional[np.ndarray] = None  # Query matrix W (stored for reference)
        self._V_pinv: Optional[np.ndarray] = None  # Cached pseudoinverse of V = B^T B

    def _build_query_matrix(self):
        """Extract (W, weights) from self.queries."""
        query_rows = []
        weight_list = []
        for q in self.queries:
            q_vec = q.query_vector()
            if q_vec.ndim > 1:
                for i in range(q_vec.shape[0]):
                    query_rows.append(q_vec[i])
                    weight_list.append(q.weight)
            else:
                query_rows.append(q_vec)
                weight_list.append(q.weight)
        if not query_rows:
            return None, None
        return np.vstack(query_rows), np.array(weight_list)

    # ------------------------------------------------------------------
    # Fourier warm-start helper (builds V_init from DFT basis)
    # ------------------------------------------------------------------

    def _fourier_warm_start(self, W, weights, n, target):
        """
        Build Fourier warm-start V_init for L-BFGS-B optimization.

        Constructs the DFT basis for the partition shape, filters conjugate-
        duplicate rows, computes optimal Fourier theta params, then builds
        V^{-1} = Re(basisT @ diag(θ) @ basisT^H) and inverts.
        """
        shape = tuple(self.domains[a] for a in self.attributes)
        if not shape:
            return np.full((n, n), target) * np.eye(n)

        # Build full DFT basis (Kronecker of per-dimension DFT matrices)
        basis = np.array([[1.0 + 0j]])
        for s in shape:
            x = np.arange(s).reshape(s, 1)
            F_s = np.exp((x @ x.T / s) * (-2j * np.pi))
            basis = np.kron(basis, F_s)
        N = basis.shape[0]

        # Reverse basis = conj(basis)^T / N
        reverse = np.conj(basis.T) / N

        # Filter conjugate-duplicate rows
        def do_filter(mat, rescale, tol=1e-6):
            prod = np.abs(mat @ mat.T)
            idx = np.where(prod > tol)
            pairs = list(zip(*idx))
            lower = sorted([x for (x, y) in pairs if x < y])
            upper = [x for (x, y) in pairs if x > y]
            reals = sorted(set(range(mat.shape[0])).difference(lower + upper))
            out = mat.copy()
            if rescale:
                out[lower] = out[lower] * 2
            return out[reals + lower]

        # Filtered reverse basis: shape (n, num_filtered)
        basisT = do_filter(reverse.T, rescale=True).T

        # Compute Fourier theta via select logic
        tmp = W @ basisT[:n, :]
        coefficients = (np.diag(weights) @ np.real(tmp * np.conj(tmp))).sum(axis=0)
        coefficients = np.maximum(coefficients, 1e-10)
        tot = sum(np.sqrt(c) for c in coefficients)
        params = tot / np.sqrt(coefficients)

        # V^{-1} = Re(basisT @ diag(θ) @ basisT^H)
        bt = basisT[:n, :]
        Vinv = np.real(bt @ np.diag(params) @ bt.conj().T)
        V_init = np.linalg.pinv(Vinv)

        # Fix diagonal and ensure PSD
        np.fill_diagonal(V_init, target)
        eig_v, vec_v = np.linalg.eigh(V_init)
        eig_v = np.maximum(eig_v, 0)
        V_init = vec_v @ np.diag(eig_v) @ vec_v.T
        np.fill_diagonal(V_init, target)
        return V_init

    # ------------------------------------------------------------------
    # Optimizer: L-BFGS-B (default)
    # ------------------------------------------------------------------

    def _get_residual_basis(self):
        """Build residual subspace basis U (columns of eigenvectors with eigenvalue 1)."""
        P_factors = []
        for attr_idx in self.attributes:
            d = self.domains[attr_idx]
            P_factors.append(np.eye(d) - (1.0/d) * np.ones((d, d)))
        if not P_factors:
            return np.array([[1.0]])
        P = reduce(np.kron, P_factors)
        eigvals, eigvecs = np.linalg.eigh(P)
        return eigvecs[:, eigvals > 0.5]

    def _lbfgsb_projected_core(self, U, A, target, S_init):
        """
        L-BFGS-B in projected residual subspace.

        Optimizes ALL unique entries of S (r×r PSD) where V = U S U^T, with:
        - Objective: trace(S^{-1} A) where A = U^T W^T D W U
        - Constraint: diag(U S U^T) <= target (via log-barrier)
        - PSD enforced via Cholesky factorization

        Unlike a full-space approach which fixes diag(V) = target, here the
        constraint diag(U S U^T) <= target is non-trivial in S-space, so we
        optimize all entries (including diagonal) with a barrier penalty.

        Args:
            U: Residual subspace basis (n × r)
            A: Projected gram matrix U^T W^T D W U (r × r)
            target: Privacy cost target
            S_init: Initial S matrix (r × r)
        """
        from scipy.linalg.lapack import dpotrf, dpotri
        from scipy import optimize

        r = U.shape[1]
        if r == 0:
            return np.zeros((1, U.shape[0]))

        # Ensure S_init is PSD and feasible
        eig_s, vec_s = np.linalg.eigh(S_init)
        S_init = vec_s @ np.diag(np.maximum(eig_s, 1e-6)) @ vec_s.T
        # Rescale to be strictly feasible: max(diag(U S U^T)) < target
        V_diag = np.sum((U @ S_init) * U, axis=1)
        max_vd = np.max(V_diag)
        if max_vd > 1e-12:
            S_init *= target / max_vd * 0.9999  # tiny margin for strict barrier feasibility

        # Extract upper triangle (including diagonal) as optimization variables
        triu_idx = np.triu_indices(r)
        x0 = S_init[triu_idx]

        # Precompute mask for off-diagonal entries (gradient doubling)
        diag_mask = triu_idx[0] == triu_idx[1]
        offdiag_mask = ~diag_mask

        # Barrier parameter: controls how tightly we enforce the constraint
        mu = 0.1  # barrier weight

        state = {'S': np.zeros((r, r)), 'last_loss': np.inf,
                 'last_pure_obj': np.nan}

        def loss_and_grad(params):
            S = state['S']
            S.fill(0)
            S[triu_idx] = params
            S = S + S.T - np.diag(np.diag(S))  # symmetrize

            # PSD check via Cholesky
            zz, info0 = dpotrf(S, False, False)
            iS, info1 = dpotri(zz)
            iS = np.triu(iS) + np.triu(iS, k=1).T
            if info0 != 0 or info1 != 0:
                return state['last_loss'] * 100, np.zeros_like(params)

            # Objective: trace(S^{-1} A)
            obj = np.sum(iS * A)
            state['last_pure_obj'] = obj

            # Diagonal constraint: diag(U S U^T) <= target
            # Use log-barrier: -mu * sum(log(target - V_ii))
            US = U @ S  # (n, r)
            V_diag = np.sum(US * U, axis=1)  # diag(U S U^T)
            slack = target - V_diag
            if np.any(slack <= 0):
                return state['last_loss'] * 100, np.zeros_like(params)

            barrier = -mu * np.sum(np.log(slack))
            loss = obj + barrier

            # Gradient of objective w.r.t. S
            G_obj = -iS @ A @ iS

            # Gradient of barrier w.r.t. S
            # d/dS (-mu log(target - u_i^T S u_i)) = mu/(target - u_i^T S u_i) * u_i u_i^T
            coeff = mu / slack  # (n,)
            G_bar = U.T @ (coeff[:, None] * U)  # (r, r) — avoids n×n diag matrix

            G = G_obj + G_bar
            # Symmetrize gradient
            G = (G + G.T) / 2

            # Extract gradient for upper triangle entries
            g = G[triu_idx]
            # Off-diagonal entries appear twice (S_{ij} and S_{ji}), so double their gradient
            g[offdiag_mask] *= 2

            state['last_loss'] = loss
            return loss, g

        # Convergence history tracking
        self._convergence_history = []
        cumulative_nfev = [0]
        iter_count = [0]

        # Early stopping state
        es_check_interval = 20  # check every 20 callback iterations
        last_check_obj = [None]

        # Record initial objective (Fourier warm-start quality)
        loss_and_grad(x0)  # evaluate to set state['last_pure_obj']
        init_obj = state['last_pure_obj']
        self._convergence_history.append({
            'nfev': 0,
            'pure_obj': init_obj,
            'mu': self.mu_schedule[0],
        })
        last_check_obj[0] = init_obj

        # Run with decreasing barrier parameter for better convergence
        self._total_nfev = 0
        self._early_stopped = False
        prev_stage_obj = init_obj

        for mu_val in self.mu_schedule:
            # Check nfev budget before starting a new stage
            if self.nfev_budget is not None and cumulative_nfev[0] >= self.nfev_budget:
                self._early_stopped = True
                break

            mu = mu_val
            iter_count[0] = 0
            last_check_obj[0] = state['last_pure_obj']

            def iter_callback(xk, mu_ref=mu_val):
                iter_count[0] += 1
                if iter_count[0] % 10 == 0:
                    self._convergence_history.append({
                        'nfev': cumulative_nfev[0] + iter_count[0],
                        'pure_obj': state['last_pure_obj'],
                        'mu': mu_ref,
                    })
                # Within-stage early stopping: check every es_check_interval iters
                if (self.early_stop_rtol is not None
                        and iter_count[0] % es_check_interval == 0):
                    cur_obj = state['last_pure_obj']
                    if last_check_obj[0] > 1e-12:
                        rel_impr = (last_check_obj[0] - cur_obj) / last_check_obj[0]
                        if rel_impr < self.early_stop_rtol:
                            return True  # signal scipy to stop this stage
                    last_check_obj[0] = cur_obj

            opts = {'maxcor': self.maxcor, 'ftol': self.ftol}
            if self.nfev_budget is not None:
                remaining = self.nfev_budget - cumulative_nfev[0]
                opts['maxfun'] = max(remaining, 10)

            result = optimize.minimize(loss_and_grad, x0, jac=True,
                                       method='L-BFGS-B',
                                       callback=iter_callback,
                                       options=opts)
            x0 = result.x
            cumulative_nfev[0] += result.nfev
            self._total_nfev += result.nfev

            # Record end-of-stage value
            self._convergence_history.append({
                'nfev': cumulative_nfev[0],
                'pure_obj': state['last_pure_obj'],
                'mu': mu_val,
            })

            # Between-stage early stopping
            cur_obj = state['last_pure_obj']
            if self.early_stop_rtol is not None and prev_stage_obj > 1e-12:
                rel_impr = (prev_stage_obj - cur_obj) / prev_stage_obj
                if rel_impr < self.early_stop_rtol:
                    self._early_stopped = True
                    break
            prev_stage_obj = cur_obj

        # Reconstruct S
        S = np.zeros((r, r))
        S[triu_idx] = result.x
        S = S + S.T - np.diag(np.diag(S))

        # Final rescale to exactly hit the constraint
        V_diag = np.sum((U @ S) * U, axis=1)
        max_vdiag = np.max(V_diag)
        if max_vdiag > 1e-12:
            S *= target / max_vdiag


        return self._cholesky_B(S) @ U.T

    # ------------------------------------------------------------------
    # Optimizer: Adaptive barrier L-BFGS-B
    # ------------------------------------------------------------------

    def _adaptive_projected_core(self, U, A, target, S_init):
        """
        L-BFGS-B with adaptive barrier schedule in projected residual subspace.

        Instead of a fixed mu schedule [0.1, 0.01, 0.001], decreases mu
        geometrically and monitors the pure objective trace(S^{-1} A).
        Stops early when improvement stalls, saving time on workloads
        where coarse barrier stages are unnecessary.

        Adaptive parameters (via convex_kwargs):
            adapt_mu_start: initial mu (default 0.1)
            adapt_mu_factor: multiplicative decrease per round (default 0.3)
            adapt_mu_min: minimum mu before final round (default 0.001)
            adapt_rtol: relative improvement threshold to stop (default 0.005)
            adapt_max_rounds: max number of barrier rounds (default 10)
        """
        from scipy.linalg.lapack import dpotrf, dpotri
        from scipy import optimize

        r = U.shape[1]
        if r == 0:
            return np.zeros((1, U.shape[0]))

        # Adaptive parameters
        mu_start = self._adapt_mu_start
        mu_factor = self._adapt_mu_factor
        mu_min = self._adapt_mu_min
        rtol = self._adapt_rtol
        max_rounds = self._adapt_max_rounds

        # Ensure S_init is PSD and feasible
        eig_s, vec_s = np.linalg.eigh(S_init)
        S_init = vec_s @ np.diag(np.maximum(eig_s, 1e-6)) @ vec_s.T
        V_diag = np.sum((U @ S_init) * U, axis=1)
        max_vd = np.max(V_diag)
        if max_vd > 1e-12:
            S_init *= target / max_vd * 0.999

        triu_idx = np.triu_indices(r)
        x0 = S_init[triu_idx]

        diag_mask = triu_idx[0] == triu_idx[1]
        offdiag_mask = ~diag_mask

        mu = mu_start

        state = {'S': np.zeros((r, r)), 'last_loss': np.inf}

        def loss_and_grad(params):
            S = state['S']
            S.fill(0)
            S[triu_idx] = params
            S = S + S.T - np.diag(np.diag(S))

            zz, info0 = dpotrf(S, False, False)
            iS, info1 = dpotri(zz)
            iS = np.triu(iS) + np.triu(iS, k=1).T
            if info0 != 0 or info1 != 0:
                return state['last_loss'] * 100, np.zeros_like(params)

            obj = np.sum(iS * A)

            US = U @ S
            V_diag = np.sum(US * U, axis=1)
            slack = target - V_diag
            if np.any(slack <= 0):
                return state['last_loss'] * 100, np.zeros_like(params)

            barrier = -mu * np.sum(np.log(slack))
            loss = obj + barrier

            G_obj = -iS @ A @ iS
            coeff = mu / slack
            G_bar = U.T @ (coeff[:, None] * U)
            G = G_obj + G_bar
            G = (G + G.T) / 2
            g = G[triu_idx]
            g[offdiag_mask] *= 2

            state['last_loss'] = loss
            return loss, g

        def eval_pure_obj(params):
            """Evaluate trace(S^{-1} A) without barrier (the actual objective)."""
            S = np.zeros((r, r))
            S[triu_idx] = params
            S = S + S.T - np.diag(np.diag(S))
            zz, info0 = dpotrf(S, False, False)
            if info0 != 0:
                return float('inf')
            iS, info1 = dpotri(zz)
            if info1 != 0:
                return float('inf')
            iS = np.triu(iS) + np.triu(iS, k=1).T
            return np.sum(iS * A)

        self._total_nfev = 0
        prev_pure_obj = eval_pure_obj(x0)
        mu = mu_start
        reached_mu_min = False

        for round_i in range(max_rounds):
            result = optimize.minimize(loss_and_grad, x0, jac=True,
                                       method='L-BFGS-B',
                                       options={'maxcor': self.maxcor,
                                                'ftol': self.ftol})
            x0 = result.x
            self._total_nfev += result.nfev

            pure_obj = eval_pure_obj(x0)

            # Check convergence: did pure objective improve enough?
            if prev_pure_obj > 1e-12:
                rel_impr = (prev_pure_obj - pure_obj) / prev_pure_obj
            else:
                rel_impr = 0.0

            prev_pure_obj = pure_obj

            # Stop if: reached minimum mu AND improvement is small
            if reached_mu_min and rel_impr < rtol:
                break

            # Decrease mu
            mu = mu * mu_factor
            if mu <= mu_min:
                mu = mu_min
                reached_mu_min = True

        # Reconstruct S
        S = np.zeros((r, r))
        S[triu_idx] = result.x
        S = S + S.T - np.diag(np.diag(S))

        V_diag = np.sum((U @ S) * U, axis=1)
        max_vdiag = np.max(V_diag)
        if max_vdiag > 1e-12:
            S *= target / max_vdiag

        return self._cholesky_B(S) @ U.T

    def _optimize_adaptive(self, W, weights, target):
        """Adaptive-barrier L-BFGS-B in projected subspace, cold start."""
        U = self._get_residual_basis()
        r = U.shape[1]
        n = W.shape[1]

        if r == 0:
            return np.zeros((1, n))

        W_proj = (W * np.sqrt(weights[:, None])) @ U
        A = W_proj.T @ W_proj

        eig, vecs = np.linalg.eigh(A)
        S_init = vecs @ np.diag(np.sqrt(np.maximum(eig, 0))) @ vecs.T
        p_diag = np.prod([(1.0 - 1.0/self.domains[a]) for a in self.attributes])
        diag_val = target / p_diag if p_diag > 1e-12 else target
        max_d = np.diag(S_init).max()
        if max_d > 1e-12:
            S_init *= diag_val / max_d

        return self._adaptive_projected_core(U, A, target, S_init)

    def _optimize_adaptive_fourier(self, W, weights, target):
        """Adaptive-barrier L-BFGS-B in projected subspace, Fourier warm-start."""
        U = self._get_residual_basis()
        r = U.shape[1]
        n = W.shape[1]

        if r == 0:
            return np.zeros((1, n))

        V_init = self._fourier_warm_start(W, weights, n, target)
        S_init = U.T @ V_init @ U

        W_proj = (W * np.sqrt(weights[:, None])) @ U
        A = W_proj.T @ W_proj

        return self._adaptive_projected_core(U, A, target, S_init)

    # ------------------------------------------------------------------
    # Optimizer: Trust-NCG (Newton-CG with Hessian-vector products)
    # ------------------------------------------------------------------

    def _newton_projected_core(self, U, A, target, S_init):
        """
        Trust-region Newton-CG in projected residual subspace.

        Same problem as _lbfgsb_projected_core but uses second-order
        information (Hessian-vector products) for faster convergence
        on ill-conditioned workloads.

        The Hessian-vector product has closed form:
          Objective:  H_obj · dS = S⁻¹ dS S⁻¹ A S⁻¹ + S⁻¹ A S⁻¹ dS S⁻¹
          Barrier:    H_bar · dS = Σᵢ (μ/slackᵢ²)(uᵢᵀ dS uᵢ)(uᵢuᵢᵀ)

        Cost per hessp call: ~3 r×r matrix multiplies (O(r³)).
        """
        from scipy.linalg.lapack import dpotrf, dpotri
        from scipy import optimize

        r = U.shape[1]
        if r == 0:
            return np.zeros((1, U.shape[0]))

        # Ensure S_init is PSD and feasible
        eig_s, vec_s = np.linalg.eigh(S_init)
        S_init = vec_s @ np.diag(np.maximum(eig_s, 1e-6)) @ vec_s.T
        V_diag = np.sum((U @ S_init) * U, axis=1)
        max_vd = np.max(V_diag)
        if max_vd > 1e-12:
            S_init *= target / max_vd * 0.999

        triu_idx = np.triu_indices(r)
        x0 = S_init[triu_idx]

        diag_mask = triu_idx[0] == triu_idx[1]
        offdiag_mask = ~diag_mask

        mu = 0.1

        # State shared between loss_and_grad and hessp
        state = {
            'S_buf': np.zeros((r, r)),
            'iS': None,          # S⁻¹, cached for hessp
            'M': None,           # S⁻¹ A S⁻¹, cached for hessp
            'slack': None,       # target - diag(U S U^T), cached for hessp
            'last_loss': np.inf,
        }

        def loss_and_grad(params):
            S = state['S_buf']
            S.fill(0)
            S[triu_idx] = params
            S = S + S.T - np.diag(np.diag(S))

            # PSD check via Cholesky
            zz, info0 = dpotrf(S, False, False)
            iS, info1 = dpotri(zz)
            iS = np.triu(iS) + np.triu(iS, k=1).T
            if info0 != 0 or info1 != 0:
                state['iS'] = None
                return state['last_loss'] * 100, np.zeros_like(params)

            obj = np.sum(iS * A)

            US = U @ S
            V_diag = np.sum(US * U, axis=1)
            slack = target - V_diag
            if np.any(slack <= 0):
                state['iS'] = None
                return state['last_loss'] * 100, np.zeros_like(params)

            barrier = -mu * np.sum(np.log(slack))
            loss = obj + barrier

            # Gradient
            M = iS @ A @ iS    # = -G_obj, cached for hessp
            G_obj = -M
            coeff = mu / slack
            G_bar = U.T @ (coeff[:, None] * U)
            G = G_obj + G_bar
            G = (G + G.T) / 2
            g = G[triu_idx]
            g[offdiag_mask] *= 2

            # Cache for hessp
            state['iS'] = iS
            state['M'] = M
            state['slack'] = slack
            state['last_loss'] = loss
            return loss, g

        def hessp_fn(params, dp):
            iS = state['iS']
            slack = state['slack']
            M = state['M']

            if iS is None:
                return np.zeros_like(dp)

            # Unpack direction to symmetric matrix
            dS = np.zeros((r, r))
            dS[triu_idx] = dp
            dS = dS + dS.T - np.diag(np.diag(dS))

            # Objective: H_obj = N @ M + (N @ M)^T
            # where N = S⁻¹ dS, M = S⁻¹ A S⁻¹ (cached)
            N = iS @ dS           # O(r³)
            NM = N @ M            # O(r³)
            H_obj = NM + NM.T    # O(r²)

            # Barrier: Σᵢ (μ/slackᵢ²)(uᵢᵀ dS uᵢ) uᵢuᵢᵀ
            UdS = U @ dS                         # (n, r)
            quad = np.sum(UdS * U, axis=1)       # uᵢᵀ dS uᵢ
            coeff2 = mu / (slack ** 2)
            H_bar = U.T @ ((coeff2 * quad)[:, None] * U)  # (r, r)

            H = H_obj + H_bar
            H = (H + H.T) / 2

            h = H[triu_idx]
            h[offdiag_mask] *= 2
            return h

        self._total_nfev = 0
        for mu_val in self.mu_schedule:
            mu = mu_val
            result = optimize.minimize(
                loss_and_grad, x0, jac=True,
                method='trust-ncg',
                hessp=hessp_fn,
                options={'maxiter': 10000, 'gtol': self.ftol})
            x0 = result.x
            self._total_nfev += result.nfev

        # Reconstruct S
        S = np.zeros((r, r))
        S[triu_idx] = result.x
        S = S + S.T - np.diag(np.diag(S))

        # Final rescale
        V_diag = np.sum((U @ S) * U, axis=1)
        max_vdiag = np.max(V_diag)
        if max_vdiag > 1e-12:
            S *= target / max_vdiag

        return self._cholesky_B(S) @ U.T

    def _optimize_newton(self, W, weights, target):
        """Trust-NCG in projected subspace, cold start."""
        U = self._get_residual_basis()
        r = U.shape[1]
        n = W.shape[1]

        if r == 0:
            return np.zeros((1, n))

        W_proj = (W * np.sqrt(weights[:, None])) @ U
        A = W_proj.T @ W_proj

        # Cold-start: matrix sqrt of A, rescaled
        eig, vecs = np.linalg.eigh(A)
        S_init = vecs @ np.diag(np.sqrt(np.maximum(eig, 0))) @ vecs.T
        p_diag = np.prod([(1.0 - 1.0/self.domains[a]) for a in self.attributes])
        diag_val = target / p_diag if p_diag > 1e-12 else target
        max_d = np.diag(S_init).max()
        if max_d > 1e-12:
            S_init *= diag_val / max_d

        return self._newton_projected_core(U, A, target, S_init)

    def _optimize_newton_fourier(self, W, weights, target):
        """Trust-NCG in projected subspace, Fourier warm-start."""
        U = self._get_residual_basis()
        r = U.shape[1]
        n = W.shape[1]

        if r == 0:
            return np.zeros((1, n))

        V_init = self._fourier_warm_start(W, weights, n, target)
        S_init = U.T @ V_init @ U

        W_proj = (W * np.sqrt(weights[:, None])) @ U
        A = W_proj.T @ W_proj

        return self._newton_projected_core(U, A, target, S_init)

    # ------------------------------------------------------------------
    # Optimizer: Hybrid (Newton for early stages, L-BFGS-B for polish)
    # ------------------------------------------------------------------

    def _hybrid_projected_core(self, U, A, target, S_init):
        """
        Hybrid optimizer: trust-ncg for early barrier stages, L-BFGS-B for final.

        Trust-NCG converges fast with second-order info but can get stuck
        near the PSD/feasibility boundary at tight barrier settings.
        L-BFGS-B handles tight barriers better via line search backtracking.

        Strategy: Newton for mu >= 0.01, L-BFGS-B for mu < 0.01.
        """
        from scipy.linalg.lapack import dpotrf, dpotri
        from scipy import optimize

        r = U.shape[1]
        if r == 0:
            return np.zeros((1, U.shape[0]))

        # Ensure S_init is PSD and feasible
        eig_s, vec_s = np.linalg.eigh(S_init)
        S_init = vec_s @ np.diag(np.maximum(eig_s, 1e-6)) @ vec_s.T
        V_diag = np.sum((U @ S_init) * U, axis=1)
        max_vd = np.max(V_diag)
        if max_vd > 1e-12:
            S_init *= target / max_vd * 0.999

        triu_idx = np.triu_indices(r)
        x0 = S_init[triu_idx]

        diag_mask = triu_idx[0] == triu_idx[1]
        offdiag_mask = ~diag_mask

        mu = 0.1

        state = {
            'S_buf': np.zeros((r, r)),
            'iS': None,
            'M': None,
            'slack': None,
            'last_loss': np.inf,
        }

        def loss_and_grad(params):
            S = state['S_buf']
            S.fill(0)
            S[triu_idx] = params
            S = S + S.T - np.diag(np.diag(S))

            zz, info0 = dpotrf(S, False, False)
            iS, info1 = dpotri(zz)
            iS = np.triu(iS) + np.triu(iS, k=1).T
            if info0 != 0 or info1 != 0:
                state['iS'] = None
                return state['last_loss'] * 100, np.zeros_like(params)

            obj = np.sum(iS * A)

            US = U @ S
            V_diag = np.sum(US * U, axis=1)
            slack = target - V_diag
            if np.any(slack <= 0):
                state['iS'] = None
                return state['last_loss'] * 100, np.zeros_like(params)

            barrier = -mu * np.sum(np.log(slack))
            loss = obj + barrier

            M = iS @ A @ iS
            G_obj = -M
            coeff = mu / slack
            G_bar = U.T @ (coeff[:, None] * U)
            G = G_obj + G_bar
            G = (G + G.T) / 2
            g = G[triu_idx]
            g[offdiag_mask] *= 2

            state['iS'] = iS
            state['M'] = M
            state['slack'] = slack
            state['last_loss'] = loss
            return loss, g

        def hessp_fn(params, dp):
            iS = state['iS']
            slack = state['slack']
            M = state['M']

            if iS is None:
                return np.zeros_like(dp)

            dS = np.zeros((r, r))
            dS[triu_idx] = dp
            dS = dS + dS.T - np.diag(np.diag(dS))

            N = iS @ dS
            NM = N @ M
            H_obj = NM + NM.T

            UdS = U @ dS
            quad = np.sum(UdS * U, axis=1)
            coeff2 = mu / (slack ** 2)
            H_bar = U.T @ ((coeff2 * quad)[:, None] * U)

            H = H_obj + H_bar
            H = (H + H.T) / 2

            h = H[triu_idx]
            h[offdiag_mask] *= 2
            return h

        # Threshold: use Newton for large mu, L-BFGS-B for small mu
        NEWTON_MU_THRESHOLD = 0.01

        self._total_nfev = 0
        for mu_val in self.mu_schedule:
            mu = mu_val
            if mu_val >= NEWTON_MU_THRESHOLD:
                result = optimize.minimize(
                    loss_and_grad, x0, jac=True,
                    method='trust-ncg',
                    hessp=hessp_fn,
                    options={'maxiter': 10000, 'gtol': self.ftol})
            else:
                result = optimize.minimize(
                    loss_and_grad, x0, jac=True,
                    method='L-BFGS-B',
                    options={'maxcor': self.maxcor, 'ftol': self.ftol})
            x0 = result.x
            self._total_nfev += result.nfev

        # Reconstruct S
        S = np.zeros((r, r))
        S[triu_idx] = result.x
        S = S + S.T - np.diag(np.diag(S))

        V_diag = np.sum((U @ S) * U, axis=1)
        max_vdiag = np.max(V_diag)
        if max_vdiag > 1e-12:
            S *= target / max_vdiag

        return self._cholesky_B(S) @ U.T

    def _optimize_hybrid(self, W, weights, target):
        """Hybrid Newton+L-BFGS-B in projected subspace, cold start."""
        U = self._get_residual_basis()
        r = U.shape[1]
        n = W.shape[1]

        if r == 0:
            return np.zeros((1, n))

        W_proj = (W * np.sqrt(weights[:, None])) @ U
        A = W_proj.T @ W_proj

        eig, vecs = np.linalg.eigh(A)
        S_init = vecs @ np.diag(np.sqrt(np.maximum(eig, 0))) @ vecs.T
        p_diag = np.prod([(1.0 - 1.0/self.domains[a]) for a in self.attributes])
        diag_val = target / p_diag if p_diag > 1e-12 else target
        max_d = np.diag(S_init).max()
        if max_d > 1e-12:
            S_init *= diag_val / max_d

        return self._hybrid_projected_core(U, A, target, S_init)

    def _optimize_hybrid_fourier(self, W, weights, target):
        """Hybrid Newton+L-BFGS-B in projected subspace, Fourier warm-start."""
        U = self._get_residual_basis()
        r = U.shape[1]
        n = W.shape[1]

        if r == 0:
            return np.zeros((1, n))

        V_init = self._fourier_warm_start(W, weights, n, target)
        S_init = U.T @ V_init @ U

        W_proj = (W * np.sqrt(weights[:, None])) @ U
        A = W_proj.T @ W_proj

        return self._hybrid_projected_core(U, A, target, S_init)

    def _optimize_lbfgsb(self, W, weights, target):
        """
        L-BFGS-B in projected subspace, cold start.
        """
        U = self._get_residual_basis()
        r = U.shape[1]
        n = W.shape[1]

        if r == 0:
            return np.zeros((1, n))

        # Project queries into subspace
        W_proj = (W * np.sqrt(weights[:, None])) @ U
        A = W_proj.T @ W_proj  # r×r

        # Cold-start: matrix sqrt of A, rescaled to diag_val
        eig, vecs = np.linalg.eigh(A)
        S_init = vecs @ np.diag(np.sqrt(np.maximum(eig, 0))) @ vecs.T
        p_diag = np.prod([(1.0 - 1.0/self.domains[a]) for a in self.attributes])
        diag_val = target / p_diag if p_diag > 1e-12 else target
        max_d = np.diag(S_init).max()
        if max_d > 1e-12:
            S_init *= diag_val / max_d

        return self._lbfgsb_projected_core(U, A, target, S_init)

    # ------------------------------------------------------------------
    # Optimizer: L-BFGS-B + Fourier warm-start
    # ------------------------------------------------------------------

    def _optimize_lbfgsb_fourier(self, W, weights, target):
        """
        L-BFGS-B in projected subspace, Fourier warm-start.

        The Fourier warm-start guarantees the initial V has the same quality
        as the Fourier solver, so the optimizer can only improve from there.
        """
        U = self._get_residual_basis()
        r = U.shape[1]
        n = W.shape[1]

        if r == 0:
            return np.zeros((1, n))

        # Fourier warm-start in full space, then project to subspace
        V_init = self._fourier_warm_start(W, weights, n, target)
        S_init = U.T @ V_init @ U

        # Project queries into subspace
        W_proj = (W * np.sqrt(weights[:, None])) @ U
        A = W_proj.T @ W_proj

        return self._lbfgsb_projected_core(U, A, target, S_init)

    # ------------------------------------------------------------------
    # Optimizer: SCS SDP (existing)
    # ------------------------------------------------------------------

    def _optimize_scs(self, W, weights, target):
        """
        SDP via CVXPY/SCS in projected residual subspace.

        Reparameterizes V = U S U^T where U spans the residual subspace,
        solves the smaller SDP over S.
        """
        import cvxpy as cp

        n_queries = W.shape[0]
        marginal_size = W.shape[1]

        # Build residual subspace basis U
        P_factors = []
        for attr_idx in self.attributes:
            d = self.domains[attr_idx]
            P_factors.append(np.eye(d) - (1.0/d) * np.ones((d, d)))

        P = np.array([[1.0]]) if not P_factors else reduce(np.kron, P_factors)
        eigvals, eigvecs = np.linalg.eigh(P)
        U = eigvecs[:, eigvals > 0.5]
        rank_P = U.shape[1]

        if rank_P == 0:
            return np.zeros((1, marginal_size))

        W_weighted = W * np.sqrt(weights[:, None])
        W_proj = W_weighted @ U

        # Warm-start
        A_proj = W_proj.T @ W_proj
        eigvals_A, eigvecs_A = np.linalg.eigh(A_proj)
        eigvals_A = np.maximum(eigvals_A, 0)
        S_init = eigvecs_A @ np.diag(np.sqrt(eigvals_A)) @ eigvecs_A.T
        V_init = U @ S_init @ U.T
        max_diag = np.max(np.diag(V_init))
        if max_diag > 1e-12:
            S_init *= target / max_diag
            V_init *= target / max_diag
        np.fill_diagonal(V_init, target)
        eig_v, vec_v = np.linalg.eigh(V_init)
        eig_v = np.maximum(eig_v, 0)
        V_init = vec_v @ np.diag(eig_v) @ vec_v.T
        S_init = U.T @ V_init @ U

        # Solve SDP
        S = cp.Variable((rank_P, rank_P), PSD=True)
        S.value = S_init
        obj_expr = 0
        for i in range(n_queries):
            obj_expr += cp.matrix_frac(W_proj[i], S)
        V_full = U @ S @ U.T
        constraints = [cp.diag(V_full) <= target]
        problem = cp.Problem(cp.Minimize(obj_expr), constraints)
        problem.solve(solver=cp.SCS, verbose=False, warm_start=True,
                      max_iters=50000, eps=1e-9)

        if problem.status not in ["optimal", "optimal_inaccurate"]:
            print(f"CVXPY solver warning: {problem.status}")
            S_opt = np.eye(rank_P)
        else:
            S_opt = S.value

        S_opt += 1e-9 * np.eye(rank_P)
        try:
            L = np.linalg.cholesky(S_opt)
            B = L.T @ U.T
        except np.linalg.LinAlgError:
            eigenvalues, eigenvectors = np.linalg.eigh(S_opt)
            eigenvalues = np.maximum(eigenvalues, 0)
            B = np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T @ U.T
        return B

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cholesky_B(X):
        """Extract B from V = X via Cholesky (or eigendecomposition fallback)."""
        try:
            L = np.linalg.cholesky(X)
            return L.T
        except np.linalg.LinAlgError:
            eig, vec = np.linalg.eigh(X)
            return np.diag(np.sqrt(np.maximum(eig, 0))) @ vec.T

    # ------------------------------------------------------------------
    # Main optimize dispatch
    # ------------------------------------------------------------------

    def optimize(self, target_privacy_cost: float = 1.0):
        """
        Find optimal mechanism for this partition's workload.

        Dispatches to the selected optimizer backend.
        """
        W, weights = self._build_query_matrix()
        if W is None:
            self.is_optimized = True
            self.B = np.zeros((0, 1))
            self.Sigma = np.eye(0)
            return

        self.W = W
        marginal_size = W.shape[1]

        if self.optimizer == 'scs':
            self.B = self._optimize_scs(W, weights, target_privacy_cost)
        elif self.optimizer == 'lbfgsb':
            self.B = self._optimize_lbfgsb(W, weights, target_privacy_cost)
        elif self.optimizer == 'lbfgsb_fourier':
            self.B = self._optimize_lbfgsb_fourier(W, weights, target_privacy_cost)
        elif self.optimizer == 'newton':
            self.B = self._optimize_newton(W, weights, target_privacy_cost)
        elif self.optimizer == 'newton_fourier':
            self.B = self._optimize_newton_fourier(W, weights, target_privacy_cost)
        elif self.optimizer == 'hybrid':
            self.B = self._optimize_hybrid(W, weights, target_privacy_cost)
        elif self.optimizer == 'hybrid_fourier':
            self.B = self._optimize_hybrid_fourier(W, weights, target_privacy_cost)
        elif self.optimizer == 'adaptive':
            self.B = self._optimize_adaptive(W, weights, target_privacy_cost)
        elif self.optimizer == 'adaptive_fourier':
            self.B = self._optimize_adaptive_fourier(W, weights, target_privacy_cost)

        # Cache V_pinv = pseudoinverse of V = B^T B using projected subspace.
        # Since B = L^T U^T, we have V = U S U^T where S = U^T V U.
        # The pseudoinverse is U S^{-1} U^T (avoids inverting noise eigenvalues).
        U = self._get_residual_basis()
        S_proj = U.T @ (self.B.T @ self.B) @ U
        S_proj_inv = np.linalg.inv(S_proj + 1e-12 * np.eye(S_proj.shape[0]))
        self._V_pinv = U @ S_proj_inv @ U.T

        self.Sigma = np.eye(self.B.shape[0])
        self.is_optimized = True

    def measure(self, x_marginal: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Apply mechanism M(x̄_A') = B x̄_A' + N(0, σ² Σ).
        """
        if not self.is_optimized:
            raise RuntimeError("Must call optimize() before measure()")
        omega = self.B @ x_marginal
        if add_noise and self.noise_scale > 0:
            noise = np.random.normal(0, np.sqrt(self.noise_scale), size=omega.shape)
            omega = omega + noise
        return omega

    def reconstruct_marginal(self, omega: np.ndarray) -> np.ndarray:
        """
        Reconstruct marginal: x̂_A' = (B^T B)^+ B^T ω
        """
        if self.B is None:
            return np.zeros(1)
        return self._V_pinv @ self.B.T @ omega

    def query_variance(self, query: Query) -> float:
        """
        Variance of query answer: Var(q^T x̂) = σ² · q^T (B^T B)^+ q
        """
        if not self.is_optimized:
            raise RuntimeError("Must call optimize() before query_variance()")

        q_vec = query.query_vector()

        if q_vec.ndim > 1:
            # Batch: sum of q_i^T V_pinv q_i = trace(Q V_pinv Q^T)
            QVp = q_vec @ self._V_pinv
            total_var = np.sum(QVp * q_vec)
            return total_var * self.noise_scale
        else:
            return (q_vec @ self._V_pinv @ q_vec) * self.noise_scale

    def _restore_from_cache(self, B, V_pinv):
        """Restore solver state from cached solution (skip optimize)."""
        self.B = B
        self._V_pinv = V_pinv
        self.Sigma = np.eye(B.shape[0])
        self.is_optimized = True


class FourierSolver(LowDimSolver):
    """
    Fourier-based solver for low-dimensional partitions.

    Implements Section 7.2 of the QuerySmasher paper.

    This solver is optimal for circular-shift invariant workloads, and provides
    a fast, numerically stable alternative to the convex solver.

    Key insight from Section 7.2:
        Due to QuerySmasher's decomposition, queries in WKLoad^{⇒A'} have the
        property that their tensor representation sums to 0 along any axis.
        This means only Fourier coefficients at indices where ALL components
        are non-zero (j_i ∈ {1, ..., d_i - 1}) are relevant.

    Paper notation mapping:
        T_A' = marginal tensor on attributes A'
        F = DFT(T_A') = Fourier transform of marginal
        θ (theta) = variance parameters for tunable Fourier coefficients
        c = spectral coefficients (weighted sum of squared magnitudes)
        γ (gamma) = sum of sqrt(c) across tunable coefficients
        Z = noisy Fourier coefficients
        T̂_A' = reconstructed marginal = IDFT(Z)
        x̄_A' = flattened marginal vector
        ω (omega) = noisy Fourier measurements (complex array)

    Tunable parameter indices (Section 7.2):
        θ[j₁, ..., jₗ] is tunable if:
        1. j_i ∈ {1, ..., d_i - 1} for all i (excludes DC component)
        2. (j₁, ..., jₗ) ≤ (d₁ - j₁, ..., dₗ - jₗ) in dictionary order
           (ensures each conjugate pair is counted once)
    """

    def __init__(self, domains: List[int], attributes: Tuple[int, ...]):
        super().__init__(domains, attributes)
        # Variance parameters θ for tunable indices (dict: index tuple -> value)
        self.theta: Optional[dict] = None
        # Spectral coefficients c for tunable indices
        self.c: Optional[dict] = None
        # Cached list of tunable indices
        self._tunable_indices: Optional[List[Tuple[int, ...]]] = None
        # Marginal shape for this partition
        self._marginal_shape: Optional[Tuple[int, ...]] = None

    def _get_marginal_shape(self) -> Tuple[int, ...]:
        """Get the shape of the marginal tensor for this partition."""
        return tuple(self.domains[attr] for attr in self.attributes)

    def _get_conjugate_index(self, idx: Tuple[int, ...]) -> Tuple[int, ...]:
        """
        Get the conjugate index for a given Fourier index.

        For index (j₁, ..., jₗ), the conjugate is (d₁ - j₁, ..., dₗ - jₗ) mod d.

        Args:
            idx: Tuple of indices (j₁, ..., jₗ)

        Returns:
            Conjugate index tuple
        """
        shape = self._marginal_shape
        return tuple((shape[i] - idx[i]) % shape[i] for i in range(len(idx)))

    def _is_self_conjugate(self, idx: Tuple[int, ...]) -> bool:
        """
        Check if an index equals its conjugate.

        This happens when j_i = d_i - j_i (mod d_i) for all i,
        which means either j_i = 0 or j_i = d_i/2 (if d_i is even).

        Args:
            idx: Tuple of indices

        Returns:
            True if idx equals its conjugate
        """
        return idx == self._get_conjugate_index(idx)

    def _get_tunable_indices(self) -> List[Tuple[int, ...]]:
        """
        Get list of tunable parameter indices per Section 7.2.

        θ[j₁, ..., jₗ] is tunable if:
        1. j_i ∈ {1, ..., d_i - 1} for all i (excludes indices with any 0)
        2. (j₁, ..., jₗ) ≤ (d₁ - j₁, ..., dₗ - jₗ) in dictionary order

        Returns:
            List of tunable index tuples
        """
        if self._tunable_indices is not None:
            return self._tunable_indices

        shape = self._marginal_shape
        if not shape:
            self._tunable_indices = []
            return self._tunable_indices

        # Generate all indices where j_i ∈ {1, ..., d_i - 1}
        import itertools
        ranges = [range(1, d) for d in shape]
        all_indices = list(itertools.product(*ranges))

        # Filter to only include indices <= their conjugate in dictionary order
        tunable = []
        for idx in all_indices:
            conj = self._get_conjugate_index(idx)
            # Include if idx <= conj in dictionary order
            if idx <= conj:
                tunable.append(idx)

        self._tunable_indices = tunable
        return self._tunable_indices

    def _precompute_tunable_arrays(self):
        """Precompute index arrays and multipliers for vectorized Fourier ops."""
        tunable = self._get_tunable_indices()
        if not tunable:
            self._tunable_idx_arrays = None
            self._multipliers = None
            return
        ndim = len(self._marginal_shape)
        self._tunable_idx_arrays = tuple(
            [idx[d] for idx in tunable] for d in range(ndim)
        )
        self._multipliers = torch.tensor([
            1.0 if self._is_self_conjugate(idx) else 4.0
            for idx in tunable
        ], dtype=torch.float64)

        # Precompute conjugate index arrays and self-conjugate mask for measure()
        self._conj_idx_arrays = tuple(
            [(self._marginal_shape[d] - idx[d]) % self._marginal_shape[d]
             for idx in tunable]
            for d in range(ndim)
        )
        self._self_conj_mask = np.array([self._is_self_conjugate(idx) for idx in tunable])

    def _build_query_batch(self):
        """Build (all_rows, *shape) tensor and weights vector from self.queries.

        Avoids per-query tensor_view() calls by stacking raw query vectors
        and reshaping in one shot.
        """
        rows = []
        weights = []
        for q in self.queries:
            q_vec = q.query_vector()
            if q_vec.ndim == 1:
                rows.append(q_vec)
                weights.append(q.weight)
            else:
                for i in range(q_vec.shape[0]):
                    rows.append(q_vec[i])
                    weights.append(q.weight)
        W = np.vstack(rows)  # (total_rows, flat_dim)
        all_tensors = torch.from_numpy(W).reshape(-1, *self._marginal_shape)
        all_weights = torch.tensor(weights, dtype=torch.float64)
        return all_tensors, all_weights

    def optimize(self, target_privacy_cost: float = 1.0):
        """
        Optimize Fourier variance parameters θ (Section 7.2).

        For each tunable index j:
        1. Compute coefficient c[j] = Σ_q weight(q) * coeff(j, q)
           where coeff(j, q) = |F̂_q[j]|² if self-conjugate, else 4|F̂_q[j]|²
        2. Set θ[j] = γ / √c[j] where γ = Σ_j √c[j]

        This allocates noise variance inversely proportional to spectral importance.

        Note: target_privacy_cost is accepted for interface compatibility but
        not used directly. All partition solvers optimize at pcost=1, and
        noise rescaling is handled by rescale_noise() during assembly.
        """
        self._marginal_shape = self._get_marginal_shape()

        if not self._marginal_shape:
            # Empty shape (scalar partition for A' = ())
            self.theta = {}
            self.c = {}
            self.is_optimized = True
            return

        tunable_indices = self._get_tunable_indices()

        if not tunable_indices:
            # No tunable indices (all domains have size 1)
            self.theta = {}
            self.c = {}
            self.is_optimized = True
            return

        self._precompute_tunable_arrays()
        fft_dims = tuple(range(1, 1 + len(self._marginal_shape)))

        # Build batch tensor directly from query vectors (skip per-query tensor_view)
        all_tensors, all_weights = self._build_query_batch()

        # Batch FFT + fancy index to extract tunable coefficients
        F_hat = torch.fft.ifftn(all_tensors, dim=fft_dims)
        mag_sq = torch.abs(F_hat[(slice(None),) + tuple(self._tunable_idx_arrays)])**2

        # c[j] = sum_q weight(q) * multiplier(j) * |F̃_q[j]|²
        c_array = (all_weights[:, None] * self._multipliers[None, :] * mag_sq).sum(dim=0)
        c_array = torch.clamp(c_array, min=1e-10)

        self.c = {idx: c_array[i].item() for i, idx in enumerate(tunable_indices)}

        # θ[j] = γ / √c[j]
        gamma = torch.sqrt(c_array).sum().item()
        self.theta = {idx: gamma / np.sqrt(self.c[idx]) for idx in tunable_indices}
        self._theta_array = torch.tensor(
            [self.theta[idx] for idx in tunable_indices], dtype=torch.float64)

        self.is_optimized = True

    def measure(self, x_marginal: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """
        Apply Fourier mechanism (Section 7.2.1).

        Procedure:
        1. Compute F = DFT(T_A') where T_A' is the marginal tensor
        2. Initialize Z = 0 (zeros tensor)
        3. For each tunable index j:
           a. Let σ² = θ[j] * noise_scale
           b. Let a + bi = F[j]
           c. Add noise: â = a + N(0, σ²), b̂ = b + N(0, σ²)
           d. Set Z[j] = â + b̂i
           e. Set Z[conj(j)] = â - b̂i (conjugate symmetry)
        4. Return Z (noisy Fourier coefficients)

        Args:
            x_marginal: Marginal data vector x̄_A' (flattened)
            add_noise: If True, add Gaussian noise. If False, return noiseless measurement.

        Returns:
            omega: Noisy Fourier coefficients Z (complex array, flattened)
        """
        if not self.is_optimized:
            raise RuntimeError("Must call optimize() before measure()")

        if not self._marginal_shape:
            # Scalar partition (A' = ()): mechanism is y = total_count + N(0, noise_scale)
            result = np.array([x_marginal.sum()])
            if add_noise and self.noise_scale > 0:
                result = result + np.random.normal(0, np.sqrt(self.noise_scale))
            return result

        # Reshape x̄_A' to tensor T_A'
        T_marginal = torch.tensor(x_marginal, dtype=torch.float64).reshape(self._marginal_shape)

        # F = DFT(T_A')
        F = torch.fft.fftn(T_marginal)

        # Extract tunable coefficients as a flat array
        tunable_idx = tuple(self._tunable_idx_arrays)
        F_tunable = F[tunable_idx]  # (n_tunable,) complex
        a = F_tunable.real.numpy()
        b = F_tunable.imag.numpy()

        if add_noise and self.noise_scale > 0:
            # σ²[j] = θ[j] * noise_scale, generate all noise at once
            stds = np.sqrt(self._theta_array.numpy() * self.noise_scale)
            a = a + np.random.normal(0, 1, size=a.shape) * stds
            b = b + np.random.normal(0, 1, size=b.shape) * stds

        # Build Z: set tunable and conjugate indices in one shot
        Z = np.zeros(self._marginal_shape, dtype=np.complex128)
        Z[tunable_idx] = a + 1j * b

        conj_idx = tuple(self._conj_idx_arrays)
        # For non-self-conjugate: Z[conj] = a - bi
        # For self-conjugate: Z[conj] = Z[idx] (already set), so overwrite is harmless
        Z[conj_idx] = a - 1j * b

        return Z.flatten()

    def reconstruct_marginal(self, omega: np.ndarray) -> np.ndarray:
        """
        Reconstruct marginal from noisy Fourier measurements (Section 7.2.2).

        T̂_A' = real(IDFT(Z))

        Args:
            omega: Noisy Fourier coefficients Z from measure() (flattened)

        Returns:
            x_hat: Reconstructed marginal x̂_A' (flattened)
        """
        if not self._marginal_shape:
            # Scalar case
            return omega.real if np.iscomplexobj(omega) else omega

        # Reshape omega back to marginal shape
        Z = torch.from_numpy(omega.reshape(self._marginal_shape))

        # T̂_A' = IDFT(Z), take real part since original data is real
        T_hat = torch.fft.ifftn(Z)
        return T_hat.real.numpy().flatten()

    def query_variance(self, query: Query) -> float:
        """
        Compute variance of query answer from Fourier mechanism.

        Var(q) = Σ_j coeff(j, q) · θ[j] · σ²

        where:
        - F̂_q = IDFT(q)
        - coeff(j, q) = |F̂_q[j]|² if self-conjugate, else 4|F̂_q[j]|²
        - σ² = self.noise_scale

        Args:
            query: Query to compute variance for

        Returns:
            Variance of the query answer
        """
        if not self.is_optimized:
            raise RuntimeError("Must call optimize() before query_variance()")

        if not self._marginal_shape:
            # Scalar partition (A' = ()): Var(q) = ||q||^2 * noise_scale
            # Mechanism is y = x + N(0, noise_scale), so reconstructed x_hat = y.
            # For subquery vector q, answer = q . x_hat, Var = ||q||^2 * noise_scale.
            q_vec = query.query_vector()
            return float(np.sum(q_vec**2)) * self.noise_scale

        if not self.theta:
            return 0.0

        q_tensors = query.tensor_view(add_batch_dim=True, batch_dim_last=False)
        if q_tensors.ndim == len(self._marginal_shape):
            q_tensors = q_tensors.unsqueeze(0)

        fft_dims = tuple(range(1, 1 + len(self._marginal_shape)))
        F_hat = torch.fft.ifftn(q_tensors, dim=fft_dims)
        mag_sq = torch.abs(F_hat[(slice(None),) + tuple(self._tunable_idx_arrays)])**2

        # Var = sum over batch and tunable: multiplier * |F̃[j]|² * θ[j] * σ²
        variance = (self._multipliers[None, :] * mag_sq * self._theta_array[None, :]).sum().item()
        return variance * self.noise_scale

    def weighted_total_variance(self) -> float:
        """Compute L_A' = Σ_q weight(q) * Var(q) via a single batched FFT."""
        if not self.is_optimized:
            raise RuntimeError("Must call optimize() before weighted_total_variance()")

        if not self._marginal_shape or not self.theta:
            # Scalar or trivial partition — fall back to base
            return sum(q.weight * self.query_variance(q) for q in self.queries)

        all_tensors, all_weights = self._build_query_batch()
        fft_dims = tuple(range(1, 1 + len(self._marginal_shape)))

        F_hat = torch.fft.ifftn(all_tensors, dim=fft_dims)
        mag_sq = torch.abs(F_hat[(slice(None),) + tuple(self._tunable_idx_arrays)])**2

        # per_row_var[i] = sum_j multiplier[j] * |F̃_i[j]|² * θ[j]
        per_row_var = (self._multipliers[None, :] * mag_sq * self._theta_array[None, :]).sum(dim=1)
        # weighted total = sum_i weight[i] * var[i] * noise_scale
        total = (all_weights * per_row_var).sum().item()
        return total * self.noise_scale

    def _restore_from_cache(self, theta, c, theta_array):
        """Restore solver state from cached solution (skip optimize)."""
        self._marginal_shape = self._get_marginal_shape()
        self.theta = theta
        self.c = c
        self._theta_array = theta_array
        self._get_tunable_indices()
        self._precompute_tunable_arrays()
        self.is_optimized = True


# ======================================================================
# Kronecker-factored solvers for multi-attribute partitions
# ======================================================================

class KronConvexSolver(LowDimSolver):
    """
    Kronecker-factored convex solver for multi-attribute partitions.

    When the workload on a partition A' = (a_1,...,a_l) has Kronecker structure
    (all queries are KronQuery with matching factors), we solve l independent
    small convex subproblems instead of one large one.

    For partition with domains [d_1,...,d_l], the full residual dimension is
    r = prod(d_i - 1). With Kron factoring, we solve l problems of dimension
    r_i = d_i - 1 each.

    Math:
      - Per-factor residual basis: U_i = eigvecs of P_i = I - J/d_i
      - Per-factor gram: A_i = U_i^T (sum_q w_q F_i^T F_i) U_i
      - Solve each: min trace(S_i^{-1} A_i) s.t. diag(U_i S_i U_i^T) <= 1
      - Combined: V = kron(V_1,...,V_l), B = kron(B_1,...,B_l)
      - Privacy cost: max(diag(V)) = prod(max(diag(V_i))) = 1
    """

    def __init__(self, domains: List[int], attributes: Tuple[int, ...],
                 optimizer: str = 'lbfgsb_fourier', factor_cache: dict = None,
                 **convex_kwargs):
        super().__init__(domains, attributes)
        self.optimizer = optimizer
        self.convex_kwargs = convex_kwargs
        self.B_factors: Optional[List[np.ndarray]] = None  # Per-factor B matrices
        self.V_pinv_factors: Optional[List[np.ndarray]] = None  # Per-factor V pseudoinverses
        # Shared cache: attr_index -> (factor_gram, B_i, V_pinv_i)
        # Passed across KronConvexSolver instances to avoid re-solving
        # the same per-attribute subproblem in different partitions.
        self._factor_cache = factor_cache if factor_cache is not None else {}

    def _get_factor_residual_basis(self, d: int) -> np.ndarray:
        """Build residual basis U for a single attribute with domain size d."""
        P = np.eye(d) - (1.0 / d) * np.ones((d, d))
        eigvals, eigvecs = np.linalg.eigh(P)
        return eigvecs[:, eigvals > 0.5]

    @staticmethod
    def _rescale_to_feasible(S, U, target, d):
        """Ensure S is PSD and diag(U S U^T) <= target."""
        eig, vec = np.linalg.eigh(S)
        S = vec @ np.diag(np.maximum(eig, 1e-6)) @ vec.T
        p_diag = 1.0 - 1.0 / d
        diag_val = target / p_diag if p_diag > 1e-12 else target
        max_d = np.diag(S).max()
        if max_d > 1e-12:
            S *= diag_val / max_d
        return S

    @staticmethod
    def _eval_objective(S, A):
        """Evaluate trace(S^{-1} A) — the pure objective without barrier."""
        try:
            S_inv = np.linalg.inv(S + 1e-12 * np.eye(S.shape[0]))
            return np.sum(S_inv * A)
        except np.linalg.LinAlgError:
            return float('inf')

    def optimize(self, target_privacy_cost: float = 1.0):
        """Solve per-factor convex subproblems independently."""
        if not self.queries:
            self.is_optimized = True
            self.B_factors = []
            self.V_pinv_factors = []
            return

        n_factors = len(self.attributes)

        # Accumulate per-factor weighted gram matrices: sum_q w_q * F_i^T F_i
        factor_grams = [np.zeros((self.domains[a], self.domains[a]))
                        for a in self.attributes]

        for q in self.queries:
            if not isinstance(q, KronQuery):
                raise TypeError(f"KronConvexSolver requires KronQuery, got {type(q)}")
            grams = q.gram_factors()
            for i in range(n_factors):
                factor_grams[i] += q.weight * grams[i]

        # Solve each factor independently (with cross-partition caching)
        self.B_factors = []
        self.V_pinv_factors = []

        for i, attr in enumerate(self.attributes):
            d = self.domains[attr]
            U = self._get_factor_residual_basis(d)
            r = U.shape[1]

            if r == 0:
                self.B_factors.append(np.zeros((1, d)))
                self.V_pinv_factors.append(np.zeros((d, d)))
                continue

            # Project gram into residual subspace
            A = U.T @ factor_grams[i] @ U  # (r, r)

            # Check cache: reuse if same attribute was solved with same gram
            if attr in self._factor_cache:
                cached_A, cached_B, cached_Vp = self._factor_cache[attr]
                if cached_A.shape == A.shape and np.allclose(cached_A, A, rtol=1e-10, atol=1e-12):
                    self.B_factors.append(cached_B)
                    self.V_pinv_factors.append(cached_Vp)
                    continue

            # Warm-start 1: matrix sqrt of A, rescaled
            eig, vecs = np.linalg.eigh(A)
            S_sqrt = vecs @ np.diag(np.sqrt(np.maximum(eig, 0))) @ vecs.T
            S_sqrt = self._rescale_to_feasible(S_sqrt, U, target_privacy_cost, d)

            # Warm-start 2: from cached 1-way solution (if available)
            S_init = S_sqrt
            if attr in self._factor_cache:
                cached_A, cached_B, cached_Vp = self._factor_cache[attr]
                # Even though A differs, the cached B captures attribute structure
                S_from_cache = U.T @ (cached_B.T @ cached_B) @ U
                S_cached = self._rescale_to_feasible(S_from_cache, U, target_privacy_cost, d)
                # Pick better starting point by objective value trace(S^{-1} A)
                obj_sqrt = self._eval_objective(S_sqrt, A)
                obj_cached = self._eval_objective(S_cached, A)
                if obj_cached < obj_sqrt:
                    S_init = S_cached

            # Use a temporary convexSolver to get access to _lbfgsb_projected_core
            tmp_solver = convexSolver(self.domains, (attr,),
                                      optimizer=self.optimizer,
                                      **self.convex_kwargs)
            B_i = tmp_solver._lbfgsb_projected_core(U, A, target_privacy_cost, S_init)

            # Compute V_pinv for this factor
            S_proj = U.T @ (B_i.T @ B_i) @ U
            S_proj_inv = np.linalg.inv(S_proj + 1e-12 * np.eye(r))
            V_pinv_i = U @ S_proj_inv @ U.T

            self.B_factors.append(B_i)
            self.V_pinv_factors.append(V_pinv_i)

            # Store in cache for other partitions sharing this attribute
            self._factor_cache[attr] = (A.copy(), B_i, V_pinv_i)

        self.is_optimized = True

    def measure(self, x_marginal: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Apply Kron-factored mechanism: B x + noise where B = kron(B_1,...,B_l).

        Uses the mixed-product property: kron(A,B) vec(X) = vec(B X A^T)
        to avoid materializing the full Kronecker product.
        """
        if not self.is_optimized:
            raise RuntimeError("Must call optimize() before measure()")

        shape = tuple(self.domains[a] for a in self.attributes)
        X = x_marginal.reshape(shape)

        # Apply B factors sequentially using tensor mode products
        # kron(B_1,...,B_l) @ vec(X) = vec of applying B_i along each mode
        result = X.copy().astype(float)
        for i in range(len(self.B_factors) - 1, -1, -1):
            B_i = self.B_factors[i]
            # Apply B_i along axis i: move axis i to last, multiply, move back
            result = np.moveaxis(result, i, -1)
            orig_shape = result.shape[:-1]
            flat = result.reshape(-1, result.shape[-1])
            flat = flat @ B_i.T  # (batch, m_i)
            new_shape = orig_shape + (B_i.shape[0],)
            result = flat.reshape(new_shape)
            result = np.moveaxis(result, -1, i)

        omega = result.flatten()

        if add_noise and self.noise_scale > 0:
            noise = np.random.normal(0, np.sqrt(self.noise_scale), size=omega.shape)
            omega = omega + noise
        return omega

    def reconstruct_marginal(self, omega: np.ndarray) -> np.ndarray:
        """Reconstruct marginal: x_hat = V_pinv @ B^T @ omega.

        Uses factored computation: V_pinv = kron(V_pinv_1,...,V_pinv_l),
        B^T = kron(B_1^T,...,B_l^T).
        """
        if not self.B_factors:
            return np.zeros(1)

        # Measurement shape: (m_1, ..., m_l) where m_i = B_i.shape[0]
        meas_shape = tuple(B.shape[0] for B in self.B_factors)
        result = omega.reshape(meas_shape)

        # Apply B_i^T then V_pinv_i along each axis (= V_pinv_i @ B_i^T)
        for i in range(len(self.B_factors) - 1, -1, -1):
            recon_i = self.V_pinv_factors[i] @ self.B_factors[i].T  # (d_i, m_i)
            result = np.moveaxis(result, i, -1)
            orig_shape = result.shape[:-1]
            flat = result.reshape(-1, result.shape[-1])
            flat = flat @ recon_i.T  # (batch, d_i)
            new_shape = orig_shape + (recon_i.shape[0],)
            result = flat.reshape(new_shape)
            result = np.moveaxis(result, -1, i)

        return result.flatten()

    def query_variance(self, query: Query) -> float:
        """Compute variance for a query.

        For KronQuery: var = prod_i(q_i^T V_pinv_i q_i) * noise_scale
        where q = kron(q_1,...,q_l). Since q has multiple rows,
        total = sum over all query rows of prod_i(row_i^T V_pinv_i row_i).

        For non-KronQuery: falls back to dense computation.
        """
        if not self.is_optimized:
            raise RuntimeError("Must call optimize() before query_variance()")

        if isinstance(query, KronQuery) and self.V_pinv_factors:
            # Per-factor variances: for each factor i, compute F_i @ V_pinv_i @ F_i^T
            # The diagonal gives per-query-row variance for that factor.
            # Total variance = sum over all Kron-product query rows of prod of factor variances.
            factor_vars = []
            for i, F in enumerate(query.factors):
                # (n_i, d_i) @ (d_i, d_i) @ (d_i, n_i) = (n_i, n_i)
                # We only need the diagonal
                FVp = F @ self.V_pinv_factors[i]
                diag_i = np.sum(FVp * F, axis=1)  # (n_i,)
                factor_vars.append(diag_i)

            # Total variance = sum over all multi-index (j_1,...,j_l)
            # of prod_i factor_vars[i][j_i]
            # = prod_i sum(factor_vars[i])  — NO, that's wrong.
            # = prod(sum_j factor_vars[i][j]) — NO.
            # Actually: sum_{j1,...,jl} prod_i var_i[j_i] = prod_i (sum_j var_i[j])
            total = 1.0
            for dv in factor_vars:
                total *= np.sum(dv)

            return total * self.noise_scale

        # Fallback: materialize and compute
        q_vec = query.query_vector()
        V_pinv = reduce(np.kron, self.V_pinv_factors)
        if q_vec.ndim > 1:
            QVp = q_vec @ V_pinv
            return float(np.sum(QVp * q_vec)) * self.noise_scale
        else:
            return float(q_vec @ V_pinv @ q_vec) * self.noise_scale

    def weighted_total_variance(self) -> float:
        """Compute L_A' using factored computation."""
        if not self.is_optimized or not self.V_pinv_factors:
            return sum(q.weight * self.query_variance(q) for q in self.queries)

        total = 0.0
        for q in self.queries:
            total += q.weight * self.query_variance(q)
        return total


class KronFourierSolver(LowDimSolver):
    """
    Kronecker-factored Fourier solver for multi-attribute partitions.

    Computes spectral coefficients per-factor instead of on the full tensor.

    For query kron(q_1,...,q_l):
      FFT(kron(q_1,...,q_l))[j_1,...,j_l] = prod_i FFT(q_i)[j_i]
      |FFT[j]|² = prod_i |FFT(q_i)[j_i]|²
      c[j] = sum_q w_q * prod_i |FFT(q_i)[j_i]|²

    This means we only need per-attribute FFTs on small vectors (d_i),
    and c is computed via outer products of per-factor magnitude arrays.
    """

    def __init__(self, domains: List[int], attributes: Tuple[int, ...]):
        super().__init__(domains, attributes)
        self.theta: Optional[dict] = None
        self.c: Optional[dict] = None
        self._marginal_shape: Optional[Tuple[int, ...]] = None
        # Vectorized storage (shape = restricted marginal shape for non-DC indices)
        self._c_full: Optional[np.ndarray] = None  # shape (d_1-1,...,d_l-1)
        self._theta_full: Optional[np.ndarray] = None
        self._mult_full: Optional[np.ndarray] = None
        self._tunable_mask: Optional[np.ndarray] = None  # bool mask over (d_1-1,...,d_l-1)
        self._tunable_freqs: Optional[list] = None

    def _get_marginal_shape(self) -> Tuple[int, ...]:
        return tuple(self.domains[attr] for attr in self.attributes)

    def optimize(self, target_privacy_cost: float = 1.0):
        """Optimize Fourier parameters — fully vectorized via outer products.

        Steps:
        1. Per-factor: IFFT each factor's rows, compute |.|², sum over rows → (d_i,)
        2. Build full c tensor via reduce(outer, per_factor_mags) → (d_1,...,d_l)
        3. Restrict to non-DC indices [1:, 1:, ...] → (d_1-1,...,d_l-1)
        4. Build multiplier array and tunable mask (idx <= conj in dict order)
        5. Compute gamma, theta from c at tunable positions
        """
        self._marginal_shape = self._get_marginal_shape()

        if not self._marginal_shape or not self.queries:
            self.theta = {}
            self.c = {}
            self.is_optimized = True
            return

        shape = self._marginal_shape
        n_factors = len(shape)

        # Step 1-2: Compute c_raw = sum_q w_q * prod_i mag_q_i
        # For each query: compute per-factor IFFT magnitudes, then outer product.
        # Accumulate across queries (sum of outer products, NOT outer product of sums).
        c_raw = np.zeros(shape)
        factor_mags = [np.zeros(d) for d in shape]  # kept for query_variance compat

        for q in self.queries:
            if not isinstance(q, KronQuery):
                raise TypeError(f"KronFourierSolver requires KronQuery, got {type(q)}")
            q_factor_mags = []
            for i, F in enumerate(q.factors):
                F_hat = np.fft.ifft(F, axis=1)  # (n_rows, d_i)
                mag2 = np.abs(F_hat) ** 2
                mag_i = mag2.sum(axis=0)  # (d_i,)
                q_factor_mags.append(mag_i)
                factor_mags[i] += q.weight * mag_i

            # Per-query c contribution: w_q * outer(mag_0, mag_1, ...)
            q_c = q_factor_mags[0]
            for i in range(1, n_factors):
                q_c = np.multiply.outer(q_c, q_factor_mags[i])
            c_raw += q.weight * q_c
        # c_raw has shape (d_1, ..., d_l)

        # Step 3: Restrict to non-DC indices: j_i in {1,...,d_i-1}
        slices = tuple(slice(1, d) for d in shape)
        c_ndc = c_raw[slices]  # shape (d_1-1, ..., d_l-1)
        reduced_shape = c_ndc.shape  # (d_1-1, ..., d_l-1)

        # Step 4: Build multiplier array and tunable mask
        # multiplier[j] = 1 if j == conj(j), else 4
        # tunable: j <= conj(j) in dictionary order
        # Work in reduced index space: reduced_idx k_i maps to freq j_i = k_i + 1
        mult = np.ones(reduced_shape)
        tunable = np.ones(reduced_shape, dtype=bool)

        # Build conjugate index arrays for each dimension
        # freq j_i = k_i + 1, conj freq = (d_i - j_i) % d_i = (d_i - k_i - 1) % d_i
        # conj in reduced space: if conj_freq >= 1 then conj_k = conj_freq - 1
        # Note: j_i in {1,...,d_i-1} so conj_freq in {1,...,d_i-1} (never 0). Good.
        conj_reduced = []
        for dim_i, d in enumerate(shape):
            k = np.arange(d - 1)  # reduced indices
            j = k + 1             # actual frequencies
            conj_j = (d - j) % d  # conjugate frequencies
            conj_k = conj_j - 1   # back to reduced indices
            conj_reduced.append(conj_k)

        # Build full multi-dim conjugate array using meshgrid
        grids = np.meshgrid(*[np.arange(d - 1) for d in shape], indexing='ij')
        conj_grids = [conj_reduced[i][grids[i]] for i in range(n_factors)]

        # Determine multiplier and tunable mask
        # Compare (k_1,...,k_l) vs (conj_k_1,...,conj_k_l) lexicographically
        # is_self_conj: all k_i == conj_k_i
        is_self_conj = np.ones(reduced_shape, dtype=bool)
        for i in range(n_factors):
            is_self_conj &= (grids[i] == conj_grids[i])
        mult = np.where(is_self_conj, 1.0, 4.0)

        # Tunable: idx <= conj in dictionary order
        # Compare lexicographically: find first dimension where they differ
        # idx <= conj iff at first differing dimension, idx[i] <= conj[i]
        is_less = np.zeros(reduced_shape, dtype=bool)  # strictly less at first diff
        is_equal_so_far = np.ones(reduced_shape, dtype=bool)
        for i in range(n_factors):
            is_less |= (is_equal_so_far & (grids[i] < conj_grids[i]))
            is_equal_so_far &= (grids[i] == conj_grids[i])
        # tunable = strictly_less OR all_equal (self-conjugate)
        tunable = is_less | is_equal_so_far

        self._mult_full = mult
        self._tunable_mask = tunable
        self._c_ndc = c_ndc  # raw c without multiplier, shape (d_1-1,...,d_l-1)

        # Step 5: c = mult * c_raw at tunable positions
        c_with_mult = mult * c_ndc
        c_tunable = c_with_mult[tunable]
        c_tunable = np.maximum(c_tunable, 1e-10)

        # theta[j] = gamma / sqrt(c[j])
        gamma = np.sum(np.sqrt(c_tunable))
        theta_tunable = gamma / np.sqrt(c_tunable)

        # Store full arrays for measure/reconstruct (theta at all positions)
        # theta_full[k] = theta at tunable position, or 0
        theta_full = np.zeros(reduced_shape)
        theta_full[tunable] = theta_tunable
        self._theta_full = theta_full

        c_full = np.zeros(reduced_shape)
        c_full[tunable] = c_tunable
        self._c_full = c_full

        # Also build index arrays for measure (tunable positions in reduced space)
        tunable_positions = np.argwhere(tunable)  # (n_tunable, n_factors)
        self._tunable_freqs = [tuple(int(p) + 1 for p in pos) for pos in tunable_positions]
        self._tunable_reduced = tunable_positions  # for indexing into reduced arrays
        self._theta_array_np = theta_tunable
        self._c_array_np = c_tunable

        # Build conjugate index arrays for tunable positions (for measure)
        self._tunable_conj_freqs = []
        for pos in tunable_positions:
            freq = tuple(int(p) + 1 for p in pos)
            conj_freq = tuple((shape[k] - freq[k]) % shape[k] for k in range(n_factors))
            self._tunable_conj_freqs.append(conj_freq)
        self._tunable_self_conj = np.array([
            f == c for f, c in zip(self._tunable_freqs, self._tunable_conj_freqs)])

        # Dict interfaces (for compatibility)
        self.c = {self._tunable_freqs[i]: c_tunable[i]
                  for i in range(len(self._tunable_freqs))}
        self.theta = {self._tunable_freqs[i]: theta_tunable[i]
                      for i in range(len(self._tunable_freqs))}

        # Cache per-factor magnitude arrays for query_variance
        self._factor_mags = factor_mags

        self.is_optimized = True

    def measure(self, x_marginal: np.ndarray, add_noise: bool = True) -> np.ndarray:
        """Apply Fourier mechanism. Vectorized over tunable indices."""
        if not self.is_optimized:
            raise RuntimeError("Must call optimize() before measure()")

        if not self._marginal_shape:
            result = np.array([x_marginal.sum()])
            if add_noise and self.noise_scale > 0:
                result = result + np.random.normal(0, np.sqrt(self.noise_scale))
            return result

        shape = self._marginal_shape
        T = x_marginal.reshape(shape)
        F = np.fft.fftn(T)
        Z = np.zeros(shape, dtype=np.complex128)

        # Extract all tunable Fourier coefficients at once
        freq_idx = tuple(np.array([f[k] for f in self._tunable_freqs])
                         for k in range(len(shape)))
        vals = F[freq_idx]  # (n_tunable,) complex
        a = vals.real.copy()
        b = vals.imag.copy()

        if add_noise and self.noise_scale > 0:
            stds = np.sqrt(self._theta_array_np * self.noise_scale)
            a += np.random.normal(0, 1, size=a.shape) * stds
            b += np.random.normal(0, 1, size=b.shape) * stds

        # Set tunable positions
        Z[freq_idx] = a + 1j * b

        # Set conjugate positions
        conj_idx = tuple(np.array([c[k] for c in self._tunable_conj_freqs])
                         for k in range(len(shape)))
        Z[conj_idx] = a - 1j * b

        return Z.flatten()

    def reconstruct_marginal(self, omega: np.ndarray) -> np.ndarray:
        """Reconstruct marginal from noisy Fourier measurements."""
        if not self._marginal_shape:
            return omega.real if np.iscomplexobj(omega) else omega
        Z = omega.reshape(self._marginal_shape)
        T_hat = np.fft.ifftn(Z)
        return T_hat.real.flatten()

    def query_variance(self, query: Query) -> float:
        """Compute variance for a query — vectorized."""
        if not self.is_optimized:
            raise RuntimeError("Must call optimize() before query_variance()")

        if not self._marginal_shape or not self.theta:
            q_vec = query.query_vector()
            return float(np.sum(q_vec ** 2)) * self.noise_scale

        if isinstance(query, KronQuery):
            # Per-factor IFFT row-summed magnitudes: (d_i,) per factor
            per_factor_sums = []
            for F in query.factors:
                F_hat = np.fft.ifft(F, axis=1)
                per_factor_sums.append((np.abs(F_hat) ** 2).sum(axis=0))

            # Build full c_q tensor via outer product → (d_1,...,d_l)
            c_q = per_factor_sums[0]
            for i in range(1, len(per_factor_sums)):
                c_q = np.multiply.outer(c_q, per_factor_sums[i])

            # Restrict to non-DC, apply multiplier and theta, sum over tunable
            slices = tuple(slice(1, d) for d in self._marginal_shape)
            c_q_ndc = c_q[slices]
            var = np.sum(c_q_ndc * self._mult_full * self._theta_full)
            return var * self.noise_scale

        # Fallback for non-KronQuery
        q_vec = query.query_vector()
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)
        q_tensors = torch.tensor(q_vec).reshape(-1, *self._marginal_shape)
        fft_dims = tuple(range(1, 1 + len(self._marginal_shape)))
        F_hat = torch.fft.ifftn(q_tensors, dim=fft_dims)

        total_var = 0.0
        for idx_i, freq in enumerate(self._tunable_freqs):
            theta_j = self._theta_array_np[idx_i]
            shape = self._marginal_shape
            conj = tuple((shape[k] - freq[k]) % shape[k] for k in range(len(freq)))
            mult = 1.0 if freq == conj else 4.0
            mag2 = torch.abs(F_hat[(slice(None),) + tuple([f] for f in freq)]) ** 2
            total_var += mult * mag2.sum().item() * theta_j

        return total_var * self.noise_scale

    def weighted_total_variance(self) -> float:
        """Compute L_A' = sum_j c[j] * theta[j] * noise_scale."""
        if not self.is_optimized or not self.theta:
            return sum(q.weight * self.query_variance(q) for q in self.queries)

        total = np.sum(self._c_array_np * self._theta_array_np)
        return total * self.noise_scale
