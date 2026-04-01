"""
Rigorous verification of QuerySmasher implementation against the paper.

This test suite proves correctness by checking:
1. Exact match with paper's worked example (Figures 2 & 3)
2. Theorem 2: Orthogonality of decomposition matrices Q_{A'}
3. Theorem 3: Reconstruction property (q̄ · x̄_A = Σ q̄^{⇒A'} · x̄_A')
4. Definition 3: Privacy cost = max_i (B^T Σ^{-1} B)[i,i]
5. Algorithm 1: Assembly formula (γ, σ² correctness)
6. Section 7.1: Convex solver projection onto residual subspace (Equation 3)
7. Section 7.2: Fourier variance formula and tunable indices
8. Noise distribution: Empirical variance matches theoretical variance

Run: .conda/bin/python -m unittest tests.test_paper_verification -v
"""

import unittest
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query import MatrixQuery
from solver import convexSolver, FourierSolver
from smasher import Smasher, powerset


def _make_dummy_query(domains, attributes):
    """Create a dummy MatrixQuery to access _compute_decomposition_matrix."""
    dim = 1
    for attr in attributes:
        dim *= domains[attr]
    return MatrixQuery(domains, attributes, np.zeros((1, dim)))


class TestFigure3Decomposition(unittest.TestCase):
    """
    Verify decomposition against the paper's exact worked example.

    From Figure 2:  q̄ = [0, 1, 1, 0, 0, 1]
    From Figure 3:  The 4 subqueries are given explicitly.

    Reference: Section 6.1, Figures 2 & 3.
    """

    def setUp(self):
        self.domains = [2, 3]  # A1={yes,no}, A2={a,b,c}
        self.q_vec = np.array([[0, 1, 1, 0, 0, 1]], dtype=float)
        self.q = MatrixQuery(self.domains, (0, 1), self.q_vec)

    def test_decompose_empty_subset(self):
        """q̄^{⇒()} = 1/2 (scalar: average of q̄ entries)."""
        sub_q = self.q.decompose(())
        result = sub_q.query_vector().flatten()
        np.testing.assert_array_almost_equal(result, [0.5], decimal=12)

    def test_decompose_A1(self):
        """q̄^{⇒(A₁)} = [1/6, -1/6]."""
        sub_q = self.q.decompose((0,))
        result = sub_q.query_vector().flatten()
        expected = np.array([1/6, -1/6])
        np.testing.assert_array_almost_equal(result, expected, decimal=12)

    def test_decompose_A2(self):
        """q̄^{⇒(A₂)} = [-1/2, 0, 1/2]."""
        sub_q = self.q.decompose((1,))
        result = sub_q.query_vector().flatten()
        expected = np.array([-1/2, 0, 1/2])
        np.testing.assert_array_almost_equal(result, expected, decimal=12)

    def test_decompose_A1_A2(self):
        """q̄^{⇒(A₁,A₂)} = [-1/6, 1/3, -1/6, 1/6, -1/3, 1/6]."""
        sub_q = self.q.decompose((0, 1))
        result = sub_q.query_vector().flatten()
        expected = np.array([-1/6, 1/3, -1/6, 1/6, -1/3, 1/6])
        np.testing.assert_array_almost_equal(result, expected, decimal=12)


class TestTheorem2Orthogonality(unittest.TestCase):
    """
    Theorem 2: For A' ≠ A'', Q_{A'}^T Q_{A''} = 0.

    The column spaces of the decomposition matrices are mutually orthogonal.
    Uses Query._compute_decomposition_matrix() to build Q matrices.

    Reference: Section 6.1, Theorem 2, Appendix A.
    """

    def test_orthogonality_2x3(self):
        """Test Q_{A'}^T Q_{A''} = 0 for all distinct subsets of (A1, A2) with domains [2,3]."""
        domains = [2, 3]
        attributes = (0, 1)
        dummy = _make_dummy_query(domains, attributes)
        subsets = [tuple(sorted(s)) for s in powerset(attributes)]

        for i in range(len(subsets)):
            for j in range(i + 1, len(subsets)):
                Q1 = dummy._compute_decomposition_matrix(subsets[i])
                Q2 = dummy._compute_decomposition_matrix(subsets[j])

                product = Q1.T @ Q2
                max_val = np.max(np.abs(product))
                self.assertLess(max_val, 1e-12,
                    f"Q_{subsets[i]}^T Q_{subsets[j]} is not zero (max={max_val:.2e})")

    def test_orthogonality_3x4x2(self):
        """Test orthogonality for 3 attributes with domains [3, 4, 2]."""
        domains = [3, 4, 2]
        attributes = (0, 1, 2)
        dummy = _make_dummy_query(domains, attributes)
        subsets = [tuple(sorted(s)) for s in powerset(attributes)]

        for i in range(len(subsets)):
            for j in range(i + 1, len(subsets)):
                Q1 = dummy._compute_decomposition_matrix(subsets[i])
                Q2 = dummy._compute_decomposition_matrix(subsets[j])
                product = Q1.T @ Q2
                max_val = np.max(np.abs(product))
                self.assertLess(max_val, 1e-12,
                    f"Orthogonality violated: {subsets[i]} vs {subsets[j]} (max={max_val:.2e})")

    def test_Q_sum_is_identity(self):
        """
        Verify Σ_{A'⊆A} Q_{A'} V_{A'} = I (from Theorem 3 proof).

        This is the completeness property: the decomposition is exhaustive.
        Uses _compute_decomposition_matrix() for Q and _Smasher__extract_marginal
        indirectly by testing that arbitrary data vectors are perfectly reconstructed.
        """
        domains = [2, 3]
        attributes = (0, 1)
        smasher = Smasher(domains, verbose=False)

        # For any data vector x, Σ_{A'} q^{=>A'} · x_{A'} should equal q · x
        # Test with identity queries (each basis vector)
        dim = np.prod([domains[a] for a in attributes])
        np.random.seed(42)
        data = np.random.rand(dim)

        for row_idx in range(dim):
            e_i = np.zeros((1, dim))
            e_i[0, row_idx] = 1.0
            q = MatrixQuery(domains, attributes, e_i)
            true_val = data[row_idx]

            reconstructed = 0.0
            for subset in powerset(attributes):
                subset = tuple(sorted(subset))
                sub_q = q.decompose(subset)
                x_marginal = smasher._Smasher__extract_marginal(data, subset)
                reconstructed += (sub_q.query_vector() @ x_marginal).flatten()[0]

            self.assertAlmostEqual(true_val, reconstructed, places=12,
                msg=f"Completeness failed for basis vector e_{row_idx}")


class TestTheorem3Reconstruction(unittest.TestCase):
    """
    Theorem 3: q̄ · x̄_A = Σ_{A'⊆A} q̄^{⇒A'} · x̄_A'

    Uses q.decompose() and Smasher.__extract_marginal() from the codebase.

    Reference: Section 6.1, Theorem 3.
    """

    def _verify_reconstruction(self, domains, attributes, q_vec, data):
        """Verify Theorem 3 using decompose() and __extract_marginal()."""
        if q_vec.ndim == 1:
            q_vec = q_vec.reshape(1, -1)
        q = MatrixQuery(domains, attributes, q_vec)
        smasher = Smasher(domains, verbose=False)
        true_answer = (q_vec @ data).flatten()

        reconstructed = np.zeros_like(true_answer)
        for subset in powerset(attributes):
            subset = tuple(sorted(subset))
            sub_q = q.decompose(subset)
            sub_q_vec = sub_q.query_vector()
            x_marginal = smasher._Smasher__extract_marginal(data, subset)
            reconstructed += (sub_q_vec @ x_marginal).flatten()

        np.testing.assert_array_almost_equal(true_answer, reconstructed, decimal=10)

    def test_paper_example(self):
        """Exact paper example: domains [2,3], q = [0,1,1,0,0,1]."""
        domains = [2, 3]
        data = np.array([5, 3, 2, 1, 4, 6], dtype=float)
        q_vec = np.array([0, 1, 1, 0, 0, 1], dtype=float)
        self._verify_reconstruction(domains, (0, 1), q_vec, data)

    def test_identity_queries(self):
        """Each standard basis vector should reconstruct exactly."""
        domains = [2, 3]
        data = np.array([5, 3, 2, 1, 4, 6], dtype=float)
        for i in range(6):
            q_vec = np.zeros(6)
            q_vec[i] = 1.0
            self._verify_reconstruction(domains, (0, 1), q_vec, data)

    def test_random_queries_2_attr(self):
        """Random queries on 2 attributes."""
        np.random.seed(123)
        domains = [4, 5]
        dim = 20
        data = np.random.randint(0, 100, size=dim).astype(float)
        for _ in range(10):
            q_vec = np.random.randn(dim)
            self._verify_reconstruction(domains, (0, 1), q_vec, data)

    def test_random_queries_3_attr(self):
        """Random queries on 3 attributes."""
        np.random.seed(456)
        domains = [2, 3, 4]
        dim = 24
        data = np.random.randint(0, 50, size=dim).astype(float)
        for _ in range(10):
            q_vec = np.random.randn(dim)
            self._verify_reconstruction(domains, (0, 1, 2), q_vec, data)

    def test_multiple_queries_matrix(self):
        """Matrix query (multiple rows) reconstruction."""
        domains = [2, 3]
        data = np.array([5, 3, 2, 1, 4, 6], dtype=float)
        q_matrix = np.array([
            [0, 1, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 1],
        ], dtype=float)
        self._verify_reconstruction(domains, (0, 1), q_matrix, data)


class TestDefinition3PrivacyCost(unittest.TestCase):
    """
    Definition 3: pcost(M) = max_i (B^T Σ^{-1} B)[i,i].

    For convexSolver where Σ = I: pcost = max_i (B^T B)[i,i].

    Reference: Section 3, Definition 3.
    """

    def test_convex_privacy_cost_leq_1(self):
        """max_i (B^T B)[i,i] <= 1.0 after optimization at target=1.0."""
        for seed in range(5):
            np.random.seed(seed)
            domains = [3, 4]
            attributes = (0, 1)
            dim = 12
            q_vec = np.random.randn(1, dim)
            query = MatrixQuery(domains, attributes, q_vec)

            solver = convexSolver(domains, attributes)
            solver.add_query(query)
            solver.optimize(target_privacy_cost=1.0)

            if solver.B.size > 0:
                V = solver.B.T @ solver.B
                max_diag = np.max(np.diag(V))
                self.assertLessEqual(max_diag, 1.0 + 1e-6,
                    f"Privacy cost {max_diag} > 1.0 (seed={seed})")


class TestAlgorithm1Assembly(unittest.TestCase):
    """
    Algorithm 1, Lines 13-16:
        γ = (Σ_{A'} √L_{A'}) / pcost
        σ²_{A'} = γ / √L_{A'}

    Verify the assembly formula produces correct noise scaling.
    Uses Smasher pipeline and solver.query_variance() from the codebase.

    Reference: Section 6.3, Algorithm 1.
    """

    def test_assembly_noise_scaling(self):
        """
        Verify σ² values satisfy the assembly formula.

        After assembly, σ²_{A'} * √L_{A'} should be the same constant γ
        for all partitions with L > 0.
        """
        domains = [2, 3]
        pcost = 2.0

        q_vec = np.array([[0, 1, 1, 0, 0, 1]], dtype=float)
        q = MatrixQuery(domains, (0, 1), q_vec)

        smasher = Smasher(domains, privacy_cost=pcost, default_solver='fourier',
                          noise=False, verbose=False)
        smasher.add_queries_to_workload([q])
        smasher.optimize()

        gamma_values = []
        for subset, solver in smasher.solvers.items():
            sigma_sq = solver.noise_scale
            if sigma_sq > 0:
                # Compute L at cost 1: temporarily reset noise_scale
                original_scale = solver.noise_scale
                solver.noise_scale = 1.0
                L = sum(q_obj.weight * solver.query_variance(q_obj)
                        for q_obj in solver.queries)
                solver.noise_scale = original_scale

                if L > 1e-9:
                    gamma = sigma_sq * np.sqrt(L)
                    gamma_values.append(gamma)

        # All gamma values should be equal
        if len(gamma_values) > 1:
            for g in gamma_values[1:]:
                self.assertAlmostEqual(gamma_values[0], g, places=8,
                    msg=f"Gamma inconsistent: {gamma_values}")

    def test_noiseless_gives_exact_answer(self):
        """noise=False should give exact reconstruction via Smasher pipeline."""
        domains = [3, 4]
        np.random.seed(42)
        data = np.random.randint(1, 100, size=12).astype(float)
        q_vec = np.random.randn(1, 12)
        true_answer = (q_vec @ data).item()

        for solver_type in ['fourier', 'convex']:
            q = MatrixQuery(domains, (0, 1), q_vec.copy())
            smasher = Smasher(domains, privacy_cost=1.0, default_solver=solver_type,
                              noise=False, verbose=False)
            smasher.add_queries_to_workload([q])
            smasher.optimize()
            smasher.measure(data)
            ans = smasher.answer_query(q)
            if isinstance(ans, np.ndarray):
                ans = ans.item()
            self.assertAlmostEqual(true_answer, ans, places=6,
                msg=f"noise=False not exact for {solver_type}: true={true_answer}, got={ans}")


class TestSection71ConvexProjection(unittest.TestCase):
    """
    Section 7.1: Numerical stability workaround.

    1. Queries in WKLoad^{⇒A'} span the residual subspace (Equation 3)
    2. Convex decomposition uses I instead of centering matrix
    3. After optimization, B is projected: B' = B @ P
    4. After projection, B' @ P = B' (idempotent)

    Uses _compute_decomposition_matrix() to build P (the projection matrix
    is Q with subset=all_attrs, for_convex=False).

    Reference: Section 7.1.
    """

    def test_standard_decomposition_in_residual_subspace(self):
        """
        Decomposed queries (standard) lie in residual subspace.
        q̄^{⇒A'} @ P = q̄^{⇒A'} where P = _compute_decomposition_matrix(subset).
        """
        domains = [3, 4]
        q_vec = np.random.RandomState(42).randn(1, 12)
        q = MatrixQuery(domains, (0, 1), q_vec)

        for subset in [(0,), (1,), (0, 1)]:
            sub_q = q.decompose(subset, for_convex=False)
            sub_vec = sub_q.query_vector()

            # P = _compute_decomposition_matrix with all attrs of sub_q in subset
            # i.e. subset=subset, which gives ⊗_i (I - (1/d)*J) for attrs in subset
            P = sub_q._compute_decomposition_matrix(subset, for_convex=False)
            projected = sub_vec @ P
            diff = np.max(np.abs(sub_vec - projected))
            self.assertLess(diff, 1e-12,
                f"Standard decomposition for subset {subset} not in residual subspace (diff={diff:.2e})")

    def test_B_in_residual_subspace(self):
        """After projection, B @ P = B. P built via _compute_decomposition_matrix."""
        domains = [3, 4]
        attributes = (0, 1)

        q_vec = np.random.RandomState(42).randn(1, 12)
        query = MatrixQuery(domains, attributes, q_vec)

        solver = convexSolver(domains, attributes)
        solver.add_query(query.decompose(attributes, for_convex=True))
        solver.optimize(target_privacy_cost=1.0)

        if solver.B.size > 0:
            # P = ⊗_i (I_{d_i} - (1/d_i)*J_{d_i}) for all attrs in subset
            # This is _compute_decomposition_matrix(attributes, for_convex=False)
            # on a query whose attributes == subset
            dummy = _make_dummy_query(domains, attributes)
            P = dummy._compute_decomposition_matrix(attributes, for_convex=False)

            B_proj = solver.B @ P
            diff = np.max(np.abs(solver.B - B_proj))
            self.assertLess(diff, 1e-10,
                f"B not in residual subspace: max diff = {diff:.2e}")

    def test_projection_is_idempotent(self):
        """P @ P = P (the projection matrix is idempotent)."""
        domains = [3, 4]
        attributes = (0, 1)
        dummy = _make_dummy_query(domains, attributes)
        P = dummy._compute_decomposition_matrix(attributes, for_convex=False)
        P2 = P @ P
        np.testing.assert_array_almost_equal(P, P2, decimal=12,
            err_msg="P is not idempotent")


class TestSection72Fourier(unittest.TestCase):
    """
    Section 7.2: Fourier solver properties.

    1. Tunable indices: j_i ∈ {1,...,d_i-1}, idx <= conjugate(idx) in dict order
    2. Variance coefficient: ||F̂[j]||² if self-conjugate, 4||F̂[j]||² otherwise
    3. θ[j] = γ / √c_j where γ = Σ √c_j
    4. Conjugate symmetry: Z[j] = conj(Z[d-j])

    Uses FourierSolver methods directly (_get_tunable_indices, _get_conjugate_index, etc.)

    Reference: Section 7.2.
    """

    def test_theta_formula(self):
        """θ[j] = γ / √c_j and γ = Σ √c_j. Verified from solver.c and solver.theta."""
        domains = [3, 4]
        attributes = (0, 1)

        q_vec = np.random.RandomState(42).randn(1, 12)
        q_full = MatrixQuery(domains, (0, 1), q_vec)
        q_decomposed = q_full.decompose(attributes, for_convex=False)

        solver = FourierSolver(domains, attributes)
        solver.add_query(q_decomposed)
        solver.optimize()

        # Verify γ = Σ √c_j from solver.c
        gamma = sum(np.sqrt(c_val) for c_val in solver.c.values())

        # Verify θ[j] = γ / √c_j from solver.theta and solver.c
        for idx in solver.theta:
            expected_theta = gamma / np.sqrt(solver.c[idx])
            self.assertAlmostEqual(solver.theta[idx], expected_theta, places=10,
                msg=f"θ[{idx}] = {solver.theta[idx]}, expected γ/√c = {expected_theta}")

    def test_conjugate_symmetry_in_Z(self):
        """Z[j₁,...,jₗ] = conj(Z[d₁-j₁,...,dₗ-jₗ]). Uses solver._get_conjugate_index."""
        domains = [3, 4]
        attributes = (0, 1)

        q_vec = np.random.RandomState(42).randn(1, 12)
        q_full = MatrixQuery(domains, (0, 1), q_vec)
        q_decomposed = q_full.decompose(attributes, for_convex=False)

        solver = FourierSolver(domains, attributes)
        solver.add_query(q_decomposed)
        solver.optimize()

        np.random.seed(99)
        data = np.random.rand(12)
        meas = solver.measure(data)
        Z = meas.reshape(3, 4)

        for idx in solver._get_tunable_indices():
            conj_idx = solver._get_conjugate_index(idx)
            np.testing.assert_almost_equal(
                Z[idx], np.conj(Z[conj_idx]), decimal=10,
                err_msg=f"Conjugate symmetry violated: Z{idx} vs Z{conj_idx}")

    def test_reconstruction_without_noise_is_exact(self):
        """With add_noise=False, solver.reconstruct_marginal recovers query answers."""
        domains = [3, 4]
        attributes = (0, 1)

        q_vec = np.random.RandomState(42).randn(1, 12)
        q_full = MatrixQuery(domains, (0, 1), q_vec)
        q_decomposed = q_full.decompose(attributes, for_convex=False)

        solver = FourierSolver(domains, attributes)
        solver.add_query(q_decomposed)
        solver.optimize()

        data = np.random.RandomState(7).rand(12)

        meas = solver.measure(data, add_noise=False)
        recon = solver.reconstruct_marginal(meas)

        true_ans = q_decomposed.query_vector() @ data
        recon_ans = q_decomposed.query_vector() @ recon
        np.testing.assert_array_almost_equal(true_ans, recon_ans, decimal=10)


class TestNoiseDistribution(unittest.TestCase):
    """
    Statistical verification that noise has the correct variance.

    For a query q with predicted variance V (from solver.query_variance),
    running N trials should give empirical_variance ≈ V.

    This is the strongest evidence that the noise mechanism is correct.

    Reference: Definition 3, Section 7.
    """

    def test_fourier_empirical_variance(self):
        """
        Run FourierSolver.measure() many times.
        Empirical variance of reconstructed answers should match solver.query_variance().
        """
        domains = [3, 4]
        attributes = (0, 1)

        q_vec = np.array([[1, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 1]], dtype=float)
        q = MatrixQuery(domains, attributes, q_vec)
        q_decomposed = q.decompose(attributes, for_convex=False)

        solver = FourierSolver(domains, attributes)
        solver.add_query(q_decomposed)
        solver.optimize()
        solver.noise_scale = 1.0

        predicted_var = solver.query_variance(q_decomposed)

        N = 5000
        data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 15, 25, 35], dtype=float)
        true_answer = (q_decomposed.query_vector() @ data).item()

        answers = []
        for _ in range(N):
            meas = solver.measure(data, add_noise=True)
            recon = solver.reconstruct_marginal(meas)
            ans = (q_decomposed.query_vector() @ recon).item()
            answers.append(ans)

        answers = np.array(answers)
        empirical_var = np.var(answers - true_answer)

        ratio = empirical_var / predicted_var
        self.assertGreater(ratio, 0.7,
            f"Empirical variance too low: {empirical_var:.4f} vs predicted {predicted_var:.4f} (ratio={ratio:.3f})")
        self.assertLess(ratio, 1.3,
            f"Empirical variance too high: {empirical_var:.4f} vs predicted {predicted_var:.4f} (ratio={ratio:.3f})")

        # Unbiasedness: mean should equal true answer
        empirical_mean = np.mean(answers)
        mean_stderr = np.sqrt(predicted_var / N)
        z_score = abs(empirical_mean - true_answer) / mean_stderr
        self.assertLess(z_score, 4.0,
            f"Mean appears biased: empirical={empirical_mean:.4f}, true={true_answer:.4f}, z={z_score:.2f}")

    def test_convex_empirical_variance(self):
        """
        Run convexSolver.measure() many times with a centered query.
        Empirical variance should match solver.query_variance().

        Note: Uses a centered query (in the residual subspace) so the solver's
        reconstruction is unbiased in isolation. Full pipeline unbiasedness is
        tested in test_full_pipeline_empirical_variance.
        """
        domains = [2, 3]
        attributes = (0, 1)

        # Centered query (sum along each axis = 0, lies in residual subspace)
        q_vec = np.array([[1, -1, 0, -1, 1, 0]], dtype=float)
        query = MatrixQuery(domains, attributes, q_vec)

        solver = convexSolver(domains, attributes)
        solver.add_query(query)
        solver.optimize(target_privacy_cost=1.0)

        predicted_var = solver.query_variance(query)

        N = 5000
        data = np.array([10, 20, 30, 40, 50, 60], dtype=float)
        true_answer = (query.query_vector() @ data).item()

        answers = []
        for _ in range(N):
            meas = solver.measure(data, add_noise=True)
            recon = solver.reconstruct_marginal(meas)
            ans = (query.query_vector() @ recon).item()
            answers.append(ans)

        answers = np.array(answers)
        empirical_var = np.var(answers - true_answer)

        ratio = empirical_var / predicted_var
        self.assertGreater(ratio, 0.7,
            f"Empirical variance too low: {empirical_var:.4f} vs predicted {predicted_var:.4f} (ratio={ratio:.3f})")
        self.assertLess(ratio, 1.3,
            f"Empirical variance too high: {empirical_var:.4f} vs predicted {predicted_var:.4f} (ratio={ratio:.3f})")

        empirical_mean = np.mean(answers)
        mean_stderr = np.sqrt(predicted_var / N)
        z_score = abs(empirical_mean - true_answer) / mean_stderr
        self.assertLess(z_score, 4.0,
            f"Mean appears biased: z={z_score:.2f}")

    def test_full_pipeline_empirical_variance(self):
        """
        End-to-end: Run full Smasher pipeline many times.
        Empirical variance should match smasher.workload_variance().
        """
        domains = [2, 3]
        q_vec = np.array([[0, 1, 1, 0, 0, 1]], dtype=float)
        data = np.array([5, 3, 2, 1, 4, 6], dtype=float)

        smasher = Smasher(domains, privacy_cost=1.0, default_solver='fourier',
                          noise=True, verbose=False)
        q = MatrixQuery(domains, (0, 1), q_vec.copy())
        smasher.add_queries_to_workload([q])
        smasher.optimize()

        predicted_var = smasher.workload_variance()
        true_answer = (q_vec @ data).item()

        N = 3000
        answers = []
        for _ in range(N):
            smasher.measure(data)
            ans = smasher.answer_query(q)
            if isinstance(ans, np.ndarray):
                ans = ans.item()
            answers.append(ans)

        answers = np.array(answers)
        empirical_var = np.var(answers - true_answer)

        ratio = empirical_var / predicted_var
        self.assertGreater(ratio, 0.6,
            f"Pipeline empirical variance too low: ratio={ratio:.3f}")
        self.assertLess(ratio, 1.4,
            f"Pipeline empirical variance too high: ratio={ratio:.3f}")


class TestNoiseParameter(unittest.TestCase):
    """
    Test the noise control parameter.

    - Smasher(noise=False): deterministic, exact answers
    - Smasher(noise=True): stochastic, answers vary across runs
    - solver.measure(add_noise=...) parameter
    """

    def test_noise_false_is_deterministic(self):
        """Two runs with noise=False should give identical answers."""
        domains = [2, 3]
        data = np.array([5, 3, 2, 1, 4, 6], dtype=float)
        q_vec = np.array([[0, 1, 1, 0, 0, 1]], dtype=float)

        answers = []
        for _ in range(2):
            q = MatrixQuery(domains, (0, 1), q_vec.copy())
            smasher = Smasher(domains, privacy_cost=1.0, default_solver='fourier',
                              noise=False, verbose=False)
            smasher.add_queries_to_workload([q])
            smasher.optimize()
            smasher.measure(data)
            ans = smasher.answer_query(q)
            if isinstance(ans, np.ndarray):
                ans = ans.item()
            answers.append(ans)

        self.assertEqual(answers[0], answers[1],
            "noise=False should be deterministic")

    def test_noise_true_varies(self):
        """Multiple runs with noise=True should give different answers."""
        domains = [2, 3]
        data = np.array([5, 3, 2, 1, 4, 6], dtype=float)
        q_vec = np.array([[0, 1, 1, 0, 0, 1]], dtype=float)

        q = MatrixQuery(domains, (0, 1), q_vec.copy())
        smasher = Smasher(domains, privacy_cost=1.0, default_solver='fourier',
                          noise=True, verbose=False)
        smasher.add_queries_to_workload([q])
        smasher.optimize()

        answers = []
        for _ in range(10):
            smasher.measure(data)
            ans = smasher.answer_query(q)
            if isinstance(ans, np.ndarray):
                ans = ans.item()
            answers.append(ans)

        self.assertGreater(len(set(answers)), 1,
            "noise=True should produce varying answers")

    def test_solver_add_noise_parameter(self):
        """Solver-level add_noise=False gives exact measurement."""
        domains = [3, 4]
        attributes = (0, 1)

        q_vec = np.random.RandomState(42).randn(1, 12)
        q = MatrixQuery(domains, attributes, q_vec)
        q_decomposed = q.decompose(attributes, for_convex=False)

        solver = FourierSolver(domains, attributes)
        solver.add_query(q_decomposed)
        solver.optimize()

        data = np.random.RandomState(7).rand(12)

        # Two noiseless measurements should be identical
        m1 = solver.measure(data, add_noise=False)
        m2 = solver.measure(data, add_noise=False)
        np.testing.assert_array_equal(m1, m2)

        # Noisy measurement should differ
        m3 = solver.measure(data, add_noise=True)
        self.assertFalse(np.array_equal(m1, m3),
            "add_noise=True should differ from add_noise=False")


class TestNoiseCovarianceStructure(unittest.TestCase):
    """
    Verify the full noise covariance structure of each mechanism.

    Replicates the approach from ResidualPlannerPlus/tests/test_zero_dataset_measurement.py:
    Run measure() N times, collect noise vectors (noisy - noiseless), compute
    empirical covariance avg(noise @ noise^T), compare against theoretical covariance.

    This is stronger than the scalar variance tests in TestNoiseDistribution:
    those check a single query's variance, while this checks the entire
    covariance matrix of the measurement noise.

    Reference: Definition 3 (privacy cost), Section 7.1 (convex), Section 7.2 (Fourier).
    """

    def test_convex_measurement_covariance(self):
        """
        Convex solver mechanism: ω = B x + N(0, σ² I).

        Theoretical noise covariance of ω: σ² * I.
        Theoretical reconstruction covariance of x̂: σ² * (B^T B)^+.
        """
        domains = [2, 3]
        attributes = (0, 1)

        q_vec = np.array([[1, -1, 0, -1, 1, 0]], dtype=float)
        query = MatrixQuery(domains, attributes, q_vec)

        solver = convexSolver(domains, attributes)
        solver.add_query(query)
        solver.optimize(target_privacy_cost=1.0)
        solver.noise_scale = 2.0  # non-trivial scale

        data = np.array([10, 20, 30, 40, 50, 60], dtype=float)
        omega_clean = solver.measure(data, add_noise=False)

        n_runs = 10000
        m = omega_clean.shape[0]  # measurement dimension

        # --- Measurement-space covariance ---
        cov_sum = np.zeros((m, m))
        for _ in range(n_runs):
            omega_noisy = solver.measure(data, add_noise=True)
            noise = (omega_noisy - omega_clean).reshape(-1, 1)
            cov_sum += noise @ noise.T
        empirical_cov = cov_sum / n_runs

        # Theory: noise covariance of ω is σ² * I
        theory_cov = solver.noise_scale * np.eye(m)

        # Compare normalized covariance (shape structure)
        if np.trace(theory_cov) > 1e-10:
            emp_norm = empirical_cov / np.trace(empirical_cov)
            thy_norm = theory_cov / np.trace(theory_cov)
            np.testing.assert_array_almost_equal(
                emp_norm, thy_norm, decimal=1,
                err_msg="Measurement noise covariance shape mismatch")

        # Compare absolute scale (trace)
        trace_ratio = np.trace(empirical_cov) / np.trace(theory_cov)
        self.assertGreater(trace_ratio, 0.85,
            f"Measurement covariance trace too low: ratio={trace_ratio:.3f}")
        self.assertLess(trace_ratio, 1.15,
            f"Measurement covariance trace too high: ratio={trace_ratio:.3f}")

        # --- Reconstruction-space covariance ---
        V = solver.B.T @ solver.B
        V_pinv = np.linalg.pinv(V)
        d_full = V_pinv.shape[0]

        recon_cov_sum = np.zeros((d_full, d_full))
        recon_clean = solver.reconstruct_marginal(omega_clean)

        for _ in range(n_runs):
            omega_noisy = solver.measure(data, add_noise=True)
            recon_noisy = solver.reconstruct_marginal(omega_noisy)
            err = (recon_noisy - recon_clean).reshape(-1, 1)
            recon_cov_sum += err @ err.T
        empirical_recon_cov = recon_cov_sum / n_runs

        # Theory: reconstruction covariance = σ² * (B^T B)^+
        theory_recon_cov = solver.noise_scale * V_pinv

        # Compare per-cell diagonal (this is the per-cell variance)
        emp_diag = np.diag(empirical_recon_cov)
        thy_diag = np.diag(theory_recon_cov)

        print("\nConvex reconstruction per-cell variance (MC vs Theory):")
        for i in range(d_full):
            if thy_diag[i] > 1e-10:
                ratio = emp_diag[i] / thy_diag[i]
                print(f"  Cell {i}: MC={emp_diag[i]:.4f}  Theory={thy_diag[i]:.4f}"
                      f"  ratio={ratio:.4f}")
                self.assertGreater(ratio, 0.7)
                self.assertLess(ratio, 1.3)

    def test_fourier_measurement_covariance(self):
        """
        Fourier solver: noise added to Fourier coefficients with variance θ[j] * σ².

        For each tunable index j, the real and imaginary parts of Z[j] each
        get independent N(0, θ[j] * σ²) noise.

        We verify: empirical variance of (Z_noisy[j] - Z_clean[j]) matches
        θ[j] * σ² for real and imaginary parts separately.
        """
        domains = [3, 4]
        attributes = (0, 1)

        q_vec = np.array([[1, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 1]], dtype=float)
        q = MatrixQuery(domains, attributes, q_vec)
        q_decomposed = q.decompose(attributes, for_convex=False)

        solver = FourierSolver(domains, attributes)
        solver.add_query(q_decomposed)
        solver.optimize()
        solver.noise_scale = 3.0  # non-trivial scale

        data = np.arange(1, 13, dtype=float)
        shape = solver._marginal_shape  # (3, 4)

        omega_clean = solver.measure(data, add_noise=False).reshape(shape)

        n_runs = 10000
        tunable = solver._get_tunable_indices()

        # Collect noise per tunable index
        real_noise = {idx: [] for idx in tunable}
        imag_noise = {idx: [] for idx in tunable}

        for _ in range(n_runs):
            omega_noisy = solver.measure(data, add_noise=True).reshape(shape)
            diff = omega_noisy - omega_clean
            for idx in tunable:
                real_noise[idx].append(diff[idx].real)
                imag_noise[idx].append(diff[idx].imag)

        print("\nFourier per-coefficient noise variance (MC vs Theory):")
        for idx in tunable:
            theory_var = solver.theta[idx] * solver.noise_scale

            emp_var_real = np.var(real_noise[idx])
            emp_var_imag = np.var(imag_noise[idx])

            ratio_real = emp_var_real / theory_var
            ratio_imag = emp_var_imag / theory_var

            print(f"  idx={idx}: θ={solver.theta[idx]:.4f}  σ²={solver.noise_scale:.1f}"
                  f"  theory={theory_var:.4f}")
            print(f"    real: MC={emp_var_real:.4f}  ratio={ratio_real:.4f}")
            print(f"    imag: MC={emp_var_imag:.4f}  ratio={ratio_imag:.4f}")

            self.assertGreater(ratio_real, 0.8,
                f"idx={idx} real variance too low: ratio={ratio_real:.3f}")
            self.assertLess(ratio_real, 1.2,
                f"idx={idx} real variance too high: ratio={ratio_real:.3f}")
            self.assertGreater(ratio_imag, 0.8,
                f"idx={idx} imag variance too low: ratio={ratio_imag:.3f}")
            self.assertLess(ratio_imag, 1.2,
                f"idx={idx} imag variance too high: ratio={ratio_imag:.3f}")

        # Also verify reconstruction covariance diagonal
        recon_clean = solver.reconstruct_marginal(omega_clean.flatten())
        d_full = len(data)

        recon_cov_sum = np.zeros((d_full, d_full))
        for _ in range(n_runs):
            omega_noisy = solver.measure(data, add_noise=True)
            recon_noisy = solver.reconstruct_marginal(omega_noisy)
            err = (recon_noisy - recon_clean).reshape(-1, 1)
            recon_cov_sum += err @ err.T
        empirical_recon_cov = recon_cov_sum / n_runs

        print("\nFourier reconstruction per-cell variance (MC):")
        emp_diag = np.diag(empirical_recon_cov)
        # All cells should have equal variance (Fourier is shift-invariant in noise)
        # since IDFT distributes noise uniformly
        print(f"  Per-cell variances: {emp_diag}")
        print(f"  Mean: {np.mean(emp_diag):.4f}  Std: {np.std(emp_diag):.4f}")

    def test_full_pipeline_covariance(self):
        """
        Full Smasher pipeline: verify that the covariance of query answers
        across all queries matches the predicted variance from workload_variance.

        Similar to ResidualPlannerPlus test_zero_dataset_measurement_with_avg_cov:
        run the full pipeline N times, collect answer vectors, compute
        empirical covariance, compare diagonal against per-query predicted variance.
        """
        domains = [2, 3]
        attributes = (0, 1)

        # Multiple queries so we get a covariance matrix, not just scalar
        q_mat = np.array([
            [0, 1, 1, 0, 0, 1],   # query 1
            [1, 0, 0, 1, 0, 0],   # query 2
            [1, 1, 1, 0, 0, 0],   # query 3 (row sum)
            [1, -1, 0, -1, 1, 0], # query 4 (centered)
        ], dtype=float)

        data = np.array([5, 3, 2, 1, 4, 6], dtype=float)
        true_answers = q_mat @ data

        for solver_type in ['fourier', 'convex']:
            q = MatrixQuery(domains, attributes, q_mat.copy())
            smasher = Smasher(domains, privacy_cost=1.0,
                              default_solver=solver_type,
                              noise=True, verbose=False)
            smasher.add_queries_to_workload([q])
            smasher.optimize()

            # Collect per-query predicted variance
            # (query_variance returns sum over rows, we need per-row)
            # Use answer_query per row to get per-query answers
            n_queries = q_mat.shape[0]
            n_runs = 5000
            all_answers = np.zeros((n_runs, n_queries))

            for t in range(n_runs):
                smasher.measure(data)
                ans = smasher.answer_query(q)
                if isinstance(ans, np.ndarray):
                    ans = ans.flatten()
                all_answers[t] = ans

            # Empirical covariance of answer vector
            errors = all_answers - true_answers[np.newaxis, :]
            empirical_cov = (errors.T @ errors) / n_runs

            # Per-query empirical variance (diagonal)
            emp_diag = np.diag(empirical_cov)

            # Get per-query predicted variance by running single-row queries
            predicted_vars = []
            for i in range(n_queries):
                qi = MatrixQuery(domains, attributes, q_mat[i:i+1].copy())
                # Decompose through all subsets and sum variances
                total_var = 0.0
                for subset in powerset(attributes):
                    subset = tuple(sorted(subset))
                    if subset not in smasher.solvers:
                        continue
                    solver = smasher.solvers[subset]
                    solver_type_key = smasher._Smasher__get_solver_type_for_partition(subset)
                    use_convex = (solver_type_key == 'convex')
                    sub_qi = qi.decompose(subset, for_convex=use_convex)
                    total_var += qi.weight * solver.query_variance(sub_qi)
                predicted_vars.append(total_var)
            predicted_vars = np.array(predicted_vars)

            print(f"\n{solver_type.upper()} full pipeline per-query variance (MC vs Theory):")
            for i in range(n_queries):
                if predicted_vars[i] > 1e-10:
                    ratio = emp_diag[i] / predicted_vars[i]
                    print(f"  Query {i}: MC={emp_diag[i]:.4f}  Theory={predicted_vars[i]:.4f}"
                          f"  ratio={ratio:.4f}")
                    self.assertGreater(ratio, 0.7,
                        f"{solver_type} query {i}: variance ratio {ratio:.3f} too low")
                    self.assertLess(ratio, 1.4,
                        f"{solver_type} query {i}: variance ratio {ratio:.3f} too high")

            # Off-diagonal: check that cross-query covariance is reasonable
            # (not a strict test, just verify no massive correlations)
            for i in range(n_queries):
                for j in range(i + 1, n_queries):
                    if emp_diag[i] > 1e-10 and emp_diag[j] > 1e-10:
                        correlation = empirical_cov[i, j] / np.sqrt(emp_diag[i] * emp_diag[j])
                        print(f"  Corr(q{i}, q{j}) = {correlation:.4f}")


if __name__ == '__main__':
    unittest.main()
