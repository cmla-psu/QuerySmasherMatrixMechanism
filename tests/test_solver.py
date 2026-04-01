"""
Unit tests for solver.py (Section 7 of QuerySmasher paper).

This module tests the low-dimensional solvers used by QuerySmasher:
- convexSolver: SDP-based optimizer (Section 7.1)
- FourierSolver: FFT-based mechanism for circular-shift invariant workloads (Section 7.2)

Test Classes:
-------------
1. TestSolvers: Basic functionality tests for both solvers
2. TestConvexSolverNumericalHealth: Tests for V = B^T B numerical stability (Section 7.1)
3. TestFourierSolverSection7_2: Comprehensive tests for Fourier solver (Section 7.2)

Key Mathematical Concepts:
--------------------------
- V = B^T B: The covariance matrix for privacy accounting
- Privacy cost: max_i (B^T B)[i,i] <= target
- Residual subspace: P = ⊗_i (I_{d_i} - (1/d_i)*J_{d_i}) (Equation 3)
- Variance: Var(q) = q^T V^{-1} q (requires V to be well-conditioned)

Run tests:
    python -m unittest tests.test_solver -v
"""

import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query import MatrixQuery
from solver import convexSolver, FourierSolver


# =============================================================================
# Basic Solver Tests
# =============================================================================

class TestSolvers(unittest.TestCase):
    def setUp(self):
        self.domains = [2, 2]
        self.attributes = (0, 1)
        # Simple residual query (interaction: parity)
        self.q_vec = np.array([[1, -1, -1, 1]], dtype=float)
        self.query = MatrixQuery(self.domains, self.attributes, self.q_vec)

    def test_convex_solver_simple(self):
        solver = convexSolver(self.domains, self.attributes)
        solver.add_query(self.query)
        
        # Optimize
        solver.optimize(target_privacy_cost=1.0)
        self.assertTrue(solver.is_optimized)
        self.assertIsNotNone(solver.B)
        
        # Measure
        data = np.array([10, 0, 0, 0]) # 1 record
        meas = solver.measure(data)
        self.assertEqual(meas.shape[0], solver.B.shape[0])
        
        # Reconstruct
        recon = solver.reconstruct_marginal(meas)
        self.assertEqual(recon.shape[0], 4) # marginal size
        
        # Variance
        var = solver.query_variance(self.query)
        self.assertGreater(var, 0.0)

    def test_fourier_solver_simple(self):
        solver = FourierSolver(self.domains, self.attributes)
        solver.add_query(self.query)

        solver.optimize()
        self.assertTrue(solver.is_optimized)
        self.assertIsNotNone(solver.theta)

        # Measure
        # Fourier needs input matching domain shape?
        # measure takes flat vector, reshapes internally
        data = np.array([5, 5, 0, 0])
        meas = solver.measure(data)
        # meas is complex FFT
        self.assertTrue(np.iscomplexobj(meas))

        # Reconstruct
        recon = solver.reconstruct_marginal(meas)
        self.assertEqual(recon.shape[0], 4)


# =============================================================================
# Convex Solver Numerical Health Tests (Section 7.1)
# =============================================================================

class TestConvexSolverNumericalHealth(unittest.TestCase):
    """
    Tests for convex solver numerical stability (Section 7.1).

    These tests verify that V = B^T B is well-conditioned and the solver
    produces valid results. A singular or ill-conditioned V would:
    1. Make variance computation undefined (q^T V^{-1} q)
    2. Potentially violate privacy guarantees
    3. Lead to incorrect reconstructions

    Key properties to verify:
    - V is positive semi-definite (eigenvalues >= 0)
    - V has expected rank (considering residual subspace projection)
    - Condition number is bounded
    - Privacy cost constraint max_i V[i,i] <= target is satisfied
    """

    def test_V_positive_semidefinite_small(self):
        """Test V = B^T B has non-negative eigenvalues for small domain."""
        domains = [3, 3]
        attributes = (0, 1)

        # Simple counting query
        q_vec = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=float)
        query = MatrixQuery(domains, attributes, q_vec)

        solver = convexSolver(domains, attributes)
        solver.add_query(query)
        solver.optimize(target_privacy_cost=1.0)

        if solver.B.size > 0:
            V = solver.B.T @ solver.B
            eigenvalues = np.linalg.eigvalsh(V)

            # All eigenvalues should be >= -epsilon (numerical tolerance)
            min_eigenvalue = np.min(eigenvalues)
            self.assertGreaterEqual(min_eigenvalue, -1e-10,
                f"V has negative eigenvalue: {min_eigenvalue}")

    def test_V_rank_matches_residual_subspace(self):
        """
        Test that V rank matches the residual subspace dimension.

        After projection B' = B @ P where P = ⊗_i (I - (1/d_i)*J),
        the rank should be at most prod(d_i - 1).
        """
        domains = [3, 4]
        attributes = (0, 1)

        q_vec = np.array([[1, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 1]], dtype=float)
        query = MatrixQuery(domains, attributes, q_vec)

        solver = convexSolver(domains, attributes)
        solver.add_query(query)
        solver.optimize(target_privacy_cost=1.0)

        if solver.B.size > 0:
            V = solver.B.T @ solver.B
            rank_V = np.linalg.matrix_rank(V, tol=1e-10)

            # Expected rank is at most (d1-1) * (d2-1) due to residual projection
            expected_max_rank = (domains[0] - 1) * (domains[1] - 1)

            self.assertLessEqual(rank_V, expected_max_rank,
                f"V rank {rank_V} exceeds residual subspace dim {expected_max_rank}")

    def test_V_condition_number_bounded(self):
        """
        Test that V condition number is bounded for small problems.

        For well-posed problems with few queries, condition number
        should not be astronomical. We use a relaxed threshold.
        """
        domains = [2, 2]
        attributes = (0, 1)

        # Parity query - simple and should be well-conditioned
        q_vec = np.array([[1, -1, -1, 1]], dtype=float)
        query = MatrixQuery(domains, attributes, q_vec)

        solver = convexSolver(domains, attributes)
        solver.add_query(query)
        solver.optimize(target_privacy_cost=1.0)

        if solver.B.size > 0:
            V = solver.B.T @ solver.B
            eigenvalues = np.linalg.eigvalsh(V)

            # Filter out near-zero eigenvalues (expected due to projection)
            nonzero_eigs = eigenvalues[eigenvalues > 1e-12]

            if len(nonzero_eigs) > 0:
                condition_number = np.max(nonzero_eigs) / np.min(nonzero_eigs)
                # For small well-posed problems, condition should be < 1e6
                self.assertLess(condition_number, 1e6,
                    f"V is ill-conditioned: cond={condition_number:.2e}")

    def test_privacy_cost_constraint_satisfied(self):
        """Test that max_i V[i,i] <= target_privacy_cost."""
        domains = [3, 3]
        attributes = (0, 1)
        target_pcost = 1.0

        q_vec = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=float)
        query = MatrixQuery(domains, attributes, q_vec)

        solver = convexSolver(domains, attributes)
        solver.add_query(query)
        solver.optimize(target_privacy_cost=target_pcost)

        if solver.B.size > 0:
            V = solver.B.T @ solver.B
            max_diag = np.max(np.diag(V))

            # Should satisfy constraint (with small numerical tolerance)
            self.assertLessEqual(max_diag, target_pcost + 1e-6,
                f"Privacy cost violated: max_diag={max_diag}, target={target_pcost}")

    def test_V_invertible_on_nonzero_eigenspace(self):
        """
        Test that V = B^T B is invertible on its non-null eigenspace.

        V is expected to be rank-deficient due to residual projection,
        but its non-zero eigenvalues should be well-behaved (not too small).
        This ensures variance computation q^T V^{-1} q gives meaningful results.
        """
        domains = [3, 3]
        attributes = (0, 1)

        q_vec = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=float)
        query = MatrixQuery(domains, attributes, q_vec)

        solver = convexSolver(domains, attributes)
        solver.add_query(query)
        solver.optimize(target_privacy_cost=1.0)

        if solver.B.size > 0:
            V = solver.B.T @ solver.B
            eigenvalues = np.linalg.eigvalsh(V)

            # Get non-zero eigenvalues (above numerical noise threshold)
            nonzero_eigs = eigenvalues[eigenvalues > 1e-10]

            if len(nonzero_eigs) > 0:
                # Condition number of the non-null eigenspace
                eig_condition = nonzero_eigs.max() / nonzero_eigs.min()

                # Condition number should be reasonable (< 10^8)
                self.assertLess(eig_condition, 1e8,
                    f"Non-null eigenspace condition number too high: {eig_condition:.2e}")

                # All non-zero eigenvalues should be positive
                self.assertTrue(all(e > 0 for e in nonzero_eigs),
                    f"Found non-positive eigenvalue in non-null space: {nonzero_eigs}")

    def test_variance_is_finite_and_positive(self):
        """
        Test that variance computation returns finite positive values.

        If V is singular, q^T V^{-1} q could be infinite or NaN.
        Using pinv should give finite results.
        """
        domains = [3, 3]
        attributes = (0, 1)

        q_vec = np.array([[1, 1, 1, 0, 0, 0, 0, 0, 0]], dtype=float)
        query = MatrixQuery(domains, attributes, q_vec)

        solver = convexSolver(domains, attributes)
        solver.add_query(query)
        solver.optimize(target_privacy_cost=1.0)

        variance = solver.query_variance(query)

        self.assertTrue(np.isfinite(variance),
            f"Variance is not finite: {variance}")
        self.assertGreaterEqual(variance, -1e-10,
            f"Variance is too negative: {variance}")

    def test_multiple_queries_numerical_stability(self):
        """Test numerical stability with multiple queries."""
        domains = [4, 4]
        attributes = (0, 1)

        # Multiple different queries
        queries = [
            MatrixQuery(domains, attributes,
                np.array([[1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)),
            MatrixQuery(domains, attributes,
                np.array([[0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=float)),
            MatrixQuery(domains, attributes,
                np.array([[1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]], dtype=float)),
        ]

        solver = convexSolver(domains, attributes)
        for q in queries:
            solver.add_query(q)
        solver.optimize(target_privacy_cost=1.0)

        if solver.B.size > 0:
            V = solver.B.T @ solver.B

            # Check basic properties
            eigenvalues = np.linalg.eigvalsh(V)
            min_eig = np.min(eigenvalues)

            # No large negative eigenvalues
            self.assertGreaterEqual(min_eig, -1e-8,
                f"V has significant negative eigenvalue: {min_eig}")

            # All variances should be finite
            for q in queries:
                var = solver.query_variance(q)
                self.assertTrue(np.isfinite(var),
                    f"Infinite variance for query")

    def test_B_projection_onto_residual_subspace(self):
        """
        Verify B is correctly projected onto residual subspace.

        After projection, B @ P should equal B (idempotent property).
        """
        from functools import reduce

        domains = [3, 4]
        attributes = (0, 1)

        q_vec = np.array([[1, 0, -1, 0, 0, 0, 0, 0, 0, -1, 0, 1]], dtype=float)
        query = MatrixQuery(domains, attributes, q_vec)

        solver = convexSolver(domains, attributes)
        solver.add_query(query)
        solver.optimize(target_privacy_cost=1.0)

        if solver.B.size > 0:
            # Build projection matrix P
            P_factors = []
            for attr_idx in attributes:
                d = domains[attr_idx]
                P_i = np.eye(d) - (1.0/d) * np.ones((d, d))
                P_factors.append(P_i)
            P = reduce(np.kron, P_factors)

            # B should already be projected, so B @ P ≈ B
            B_proj = solver.B @ P
            projection_error = np.max(np.abs(solver.B - B_proj))

            self.assertLess(projection_error, 1e-10,
                f"B not properly projected: error={projection_error:.2e}")

    def test_empty_query_list(self):
        """Test solver handles empty query list gracefully."""
        domains = [3, 3]
        attributes = (0, 1)

        solver = convexSolver(domains, attributes)
        # Don't add any queries
        solver.optimize(target_privacy_cost=1.0)

        self.assertTrue(solver.is_optimized)
        # B should be empty or minimal
        self.assertEqual(solver.B.size, 0)

    def test_single_element_domain(self):
        """Test solver with domain containing single element."""
        domains = [1, 3]
        attributes = (0, 1)

        q_vec = np.array([[1, 0, -1]], dtype=float)
        query = MatrixQuery(domains, attributes, q_vec)

        solver = convexSolver(domains, attributes)
        solver.add_query(query)
        solver.optimize(target_privacy_cost=1.0)

        # Should complete without error
        self.assertTrue(solver.is_optimized)


# =============================================================================
# Fourier Solver Tests (Section 7.2)
# =============================================================================

class TestFourierSolverSection7_2(unittest.TestCase):
    """
    Comprehensive tests for FourierSolver based on Section 7.2 of the paper.

    These tests verify:
    1. Tunable indices are correctly identified (non-zero components, conjugate ordering)
    2. Conjugate symmetry is properly handled in measurement
    3. Variance computation is correct (including 4x factor for non-self-conjugate)
    4. Reconstruction preserves query answers without noise
    """

    def test_tunable_indices_2x2(self):
        """Test tunable index identification for 2x2 domain."""
        domains = [2, 2]
        attributes = (0, 1)

        solver = FourierSolver(domains, attributes)
        solver._marginal_shape = (2, 2)

        tunable = solver._get_tunable_indices()

        # For 2x2, only index (1,1) has all components in {1, ..., d-1}
        # And (1,1) conjugate is (2-1, 2-1) = (1,1), so it's self-conjugate
        self.assertEqual(len(tunable), 1)
        self.assertIn((1, 1), tunable)

    def test_tunable_indices_3x3(self):
        """Test tunable index identification for 3x3 domain."""
        domains = [3, 3]
        attributes = (0, 1)

        solver = FourierSolver(domains, attributes)
        solver._marginal_shape = (3, 3)

        tunable = solver._get_tunable_indices()

        # For 3x3, indices with all components in {1, 2}:
        # (1,1), (1,2), (2,1), (2,2)
        # Conjugates: (1,1) -> (2,2), (1,2) -> (2,1)
        # Dictionary order filtering:
        # (1,1) <= (2,2)? Yes
        # (1,2) <= (2,1)? Yes (1 < 2)
        # (2,1) <= (1,2)? No (2 > 1)
        # (2,2) <= (1,1)? No (2 > 1)
        # So tunable indices should be: (1,1), (1,2)
        self.assertEqual(len(tunable), 2)
        self.assertIn((1, 1), tunable)
        self.assertIn((1, 2), tunable)
        self.assertNotIn((2, 1), tunable)
        self.assertNotIn((2, 2), tunable)

    def test_self_conjugate_check(self):
        """Test self-conjugate detection."""
        # For 4x4 domain
        domains = [4, 4]
        attributes = (0, 1)

        solver = FourierSolver(domains, attributes)
        solver._marginal_shape = (4, 4)

        # (2,2) conjugate is (4-2, 4-2) = (2,2), self-conjugate
        self.assertTrue(solver._is_self_conjugate((2, 2)))

        # (1,1) conjugate is (3,3), not self-conjugate
        self.assertFalse(solver._is_self_conjugate((1, 1)))

        # (1,2) conjugate is (3,2), not self-conjugate
        self.assertFalse(solver._is_self_conjugate((1, 2)))

    def test_conjugate_symmetry_in_measurement(self):
        """Test that measurement produces conjugate-symmetric Fourier coefficients."""
        domains = [3, 3]
        attributes = (0, 1)

        # Use a centered query (sum along axes = 0)
        # This is what queries in WKLoad^{=>A'} look like
        q_vec = np.array([[1, 0, -1, 0, 0, 0, -1, 0, 1]], dtype=float)
        query = MatrixQuery(domains, attributes, q_vec)

        solver = FourierSolver(domains, attributes)
        solver.add_query(query)
        solver.optimize()

        # Use deterministic data
        np.random.seed(42)
        data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)

        # Measure multiple times to check conjugate symmetry
        meas = solver.measure(data)
        Z = meas.reshape((3, 3))

        # Check conjugate pairs
        # Z[1,1] should be conjugate of Z[2,2]
        np.testing.assert_almost_equal(Z[1, 1], np.conj(Z[2, 2]), decimal=10)

        # Z[1,2] should be conjugate of Z[2,1]
        np.testing.assert_almost_equal(Z[1, 2], np.conj(Z[2, 1]), decimal=10)

    def test_reconstruction_without_noise(self):
        """Test that reconstruction preserves query answer when noise_scale=0."""
        domains = [3, 3]
        attributes = (0, 1)

        # Centered query
        q_vec = np.array([[1, 0, -1, 0, 0, 0, -1, 0, 1]], dtype=float)
        query = MatrixQuery(domains, attributes, q_vec)

        solver = FourierSolver(domains, attributes)
        solver.add_query(query)
        solver.optimize()
        solver.noise_scale = 0  # No noise

        data = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90], dtype=float)
        true_answer = q_vec @ data

        meas = solver.measure(data)
        recon = solver.reconstruct_marginal(meas)
        reconstructed_answer = q_vec @ recon

        np.testing.assert_almost_equal(reconstructed_answer, true_answer, decimal=10)

    def test_variance_formula_single_query(self):
        """Test variance computation for a single centered query."""
        domains = [2, 2]
        attributes = (0, 1)

        # Parity query: centered, interaction term
        q_vec = np.array([[1, -1, -1, 1]], dtype=float)
        query = MatrixQuery(domains, attributes, q_vec)

        solver = FourierSolver(domains, attributes)
        solver.add_query(query)
        solver.optimize()

        # Get variance
        var = solver.query_variance(query)

        # For this query, only (1,1) is relevant
        # Variance should be positive and finite
        self.assertGreater(var, 0)
        self.assertTrue(np.isfinite(var))

    def test_larger_domain(self):
        """Test FourierSolver on larger domain."""
        domains = [4, 5]
        attributes = (0, 1)

        # Create a centered query
        # Row sums and column sums should be 0
        q_matrix = np.array([
            [1, 0, -1, 0, 0],
            [0, 0, 0, 0, 0],
            [-1, 0, 1, 0, 0],
            [0, 0, 0, 0, 0]
        ], dtype=float).flatten().reshape(1, -1)

        query = MatrixQuery(domains, attributes, q_matrix)

        solver = FourierSolver(domains, attributes)
        solver.add_query(query)
        solver.optimize()

        self.assertTrue(solver.is_optimized)
        self.assertTrue(len(solver.theta) > 0)

        # Test measure/reconstruct cycle
        data = np.random.rand(20)
        solver.noise_scale = 0

        meas = solver.measure(data)
        recon = solver.reconstruct_marginal(meas)

        true_answer = q_matrix @ data
        reconstructed_answer = q_matrix @ recon

        np.testing.assert_almost_equal(reconstructed_answer, true_answer, decimal=10)

    def test_empty_partition(self):
        """Test FourierSolver with empty attribute set (scalar partition)."""
        domains = [3, 3]
        attributes = ()  # Empty

        solver = FourierSolver(domains, attributes)

        solver.optimize()

        self.assertTrue(solver.is_optimized)
        # theta should be empty dict for empty attributes
        self.assertEqual(len(solver.theta), 0)


if __name__ == '__main__':
    unittest.main()
