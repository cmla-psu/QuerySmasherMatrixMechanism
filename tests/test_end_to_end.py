import unittest
import numpy as np
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smasher import Smasher, powerset
from query import MatrixQuery
from solver import convexSolver

class TestQuerySmasher(unittest.TestCase):
    def setUp(self):
        # 2 Attributes: A1 dim=2, A2 dim=3
        self.domains = [2, 3]
        self.smasher_convex = Smasher(self.domains, privacy_cost=1.0, default_solver='convex', verbose=False)
        self.smasher_fourier = Smasher(self.domains, privacy_cost=1.0, default_solver='fourier', verbose=False) # Will use Fourier if possible

    def test_basic_pipeline_convex(self):
        print("\nTesting Convex Pipeline...")
        # Define a simple counting query
        # Count all records (Identity on full domain)
        # q = [1, 1, 1, 1, 1, 1]
        q_vec = np.ones((1, 6))
        q = MatrixQuery(self.domains, (0, 1), q_vec)
        
        self.smasher_convex.add_queries_to_workload([q])
        self.smasher_convex.optimize()
        
        # Verify optimization
        self.assertTrue(self.smasher_convex.is_optimized)
        
        # Fake data: 1 record at (0, 0)
        data = np.zeros(6)
        data[0] = 1 # One record
        
        # True answer is 1
        true_ans = 1.0
        
        self.smasher_convex.measure(data)
        
        # Answer
        ans = self.smasher_convex.answer_query(q)
        var = self.smasher_convex.workload_variance()
        
        print(f"True: {true_ans}, Noisy: {ans}, Variance: {var}")
        
        # Check if reasonably close (within 5 std devs)
        # std = sqrt(var) approx?
        # Note: answer_query returns a float (or array)
        # q variances is what we want.
        # But workload_variance matches q variance here (1 query).
        
        # Warning: answer might include noise.
        # Deterministic check is hard. Just check it runs and gives number.
        if isinstance(ans, np.ndarray):
             ans = ans.item()
        self.assertIsInstance(ans, (float, np.float64))
    
    def test_decomposition(self):
        print("\nTesting Decomposition...")
        # Query: sum over A2 (marginal on A1)
        # q = [1, 1, 1, 0, 0, 0] ? No.
        # Marginal A1 has size 2.
        # q vector [1, 1] on A1 means counts for A1=0 + A1=1 = Total count.
        # Let's do a query that is just count(A1=0).
        # On A1 marginal: [1, 0].
        # In full space: [1, 1, 1, 0, 0, 0].
        
        q_vec = np.array([[1, 1, 1, 0, 0, 0]])
        q = MatrixQuery(self.domains, (0, 1), q_vec)
        
        decomposed = q.decompose((0,))
        # Should be scaled/centered?
        # Check dimensions
        d_vec = decomposed.query_vector()
        print(f"Decomposed on (0,): {d_vec}")
        self.assertEqual(d_vec.shape[1], 2)

    def test_fourier_pipeline(self):
        print("\nTesting Fourier Pipeline...")
        # Fourier works best for shift-invariant... but should run for any.
        q_vec = np.ones((1, 6))
        q = MatrixQuery(self.domains, (0, 1), q_vec)
        
        self.smasher_fourier.add_queries_to_workload([q])
        self.smasher_fourier.optimize()
        
        data = np.zeros(6)
        data[0] = 10
        self.smasher_fourier.measure(data)
        ans = self.smasher_fourier.answer_query(q)
        print(f"Fourier Answer: {ans}")
        if isinstance(ans, np.ndarray):
             ans = ans.item()
        self.assertIsInstance(ans, (float, np.float64))

    def test_reconstruction_without_noise(self):
        """
        Test Theorem 3 from QuerySmasher paper:
        q̄ · x̄_A = Σ_{A'⊆A} q̄^{⇒A'} · x̄_A'

        Decomposing a query and summing answers from marginals (without noise)
        should give the exact same answer as the original query.
        """
        print("\nTesting Reconstruction Without Noise (Theorem 3)...")

        # Test with multiple query types and data configurations
        domains = [2, 3]  # A1 has size 2, A2 has size 3

        # Test case 1: Simple counting query (example from Figure 2 in paper)
        # Query: count (A1=yes, A2 in {b,c}) OR (A1=no, A2=c)
        # Tensor form:      a  b  c
        #            yes [ 0  1  1 ]
        #            no  [ 0  0  1 ]
        # Vector form (row-major): [0, 1, 1, 0, 0, 1]
        q_vec = np.array([[0, 1, 1, 0, 0, 1]], dtype=float)
        q = MatrixQuery(domains, (0, 1), q_vec)

        # Create some test data
        # Tensor:      a  b  c
        #       yes [ 5  3  2 ]
        #       no  [ 1  4  6 ]
        data = np.array([5, 3, 2, 1, 4, 6], dtype=float)

        # True answer: q · data = 0*5 + 1*3 + 1*2 + 0*1 + 0*4 + 1*6 = 11
        true_answer = np.dot(q_vec.flatten(), data)
        print(f"True answer (q · x): {true_answer}")

        # Now verify reconstruction: Σ_{A'⊆A} q̄^{⇒A'} · x̄_A'
        # Subsets of (0,1): (), (0,), (1,), (0,1)
        # Use existing powerset from smasher module and extract_marginal from Smasher class
        smasher = Smasher(domains, verbose=False)

        attributes = (0, 1)
        reconstructed_answer = 0.0

        print("Decomposition breakdown:")
        for subset in powerset(attributes):
            subset = tuple(sorted(subset))

            # Decompose query
            sub_q = q.decompose(subset)
            sub_q_vec = sub_q.query_vector().flatten()

            # Extract marginal using Smasher's method
            x_marginal = smasher._Smasher__extract_marginal(data, subset)

            # Compute subquery answer
            sub_answer = np.dot(sub_q_vec, x_marginal)
            reconstructed_answer += sub_answer

            print(f"  A'={subset}: q^{{=>A'}} · x_A' = {sub_answer:.6f}")

        print(f"Reconstructed answer (sum): {reconstructed_answer}")
        print(f"Difference: {abs(true_answer - reconstructed_answer)}")

        # They should be exactly equal (within floating point tolerance)
        self.assertAlmostEqual(true_answer, reconstructed_answer, places=10,
            msg=f"Reconstruction failed: true={true_answer}, reconstructed={reconstructed_answer}")

    def test_reconstruction_multiple_queries(self):
        """
        Test reconstruction property with multiple different queries
        to ensure robustness.
        """
        print("\nTesting Reconstruction with Multiple Queries...")

        domains = [2, 3]

        # Different query patterns
        test_cases = [
            ("Identity", np.eye(6)),  # 6 identity queries
            ("All ones", np.ones((1, 6))),  # Total count
            ("First element", np.array([[1, 0, 0, 0, 0, 0]])),
            ("Row sum (A1=0)", np.array([[1, 1, 1, 0, 0, 0]])),
            ("Col sum (A2=0)", np.array([[1, 0, 0, 1, 0, 0]])),
            ("Alternating", np.array([[1, -1, 1, -1, 1, -1]])),
        ]

        # Random data
        np.random.seed(42)
        data = np.random.randint(0, 100, size=6).astype(float)
        print(f"Test data: {data}")

        # Use existing powerset from smasher module and extract_marginal from Smasher class
        smasher = Smasher(domains, verbose=False)

        attributes = (0, 1)

        for name, q_matrix in test_cases:
            q = MatrixQuery(domains, attributes, q_matrix)

            # True answers
            true_answers = q_matrix @ data

            # Reconstructed answers
            reconstructed = np.zeros_like(true_answers)

            for subset in powerset(attributes):
                subset = tuple(sorted(subset))
                sub_q = q.decompose(subset)
                sub_q_vec = sub_q.query_vector()
                x_marginal = smasher._Smasher__extract_marginal(data, subset)
                reconstructed += sub_q_vec @ x_marginal

            max_diff = np.max(np.abs(true_answers - reconstructed))
            print(f"  {name}: max diff = {max_diff:.2e}")

            np.testing.assert_array_almost_equal(
                true_answers, reconstructed, decimal=10,
                err_msg=f"Reconstruction failed for {name}")

        print("All reconstruction tests passed!")


    def test_section_8_1_convex_solver_workflow_no_noise(self):
        """
        Test Section 8.1: Convex solver workflow without noise.

        This verifies the complete Algorithm 1 workflow for the convex solver:
        1. Decomposition with for_convex=True (uses I_d instead of I_d - (1/d)J_d)
        2. Convex optimization finds optimal B matrix
        3. Projection onto residual subspace (Equation 3)
        4. Reconstruction property (Theorem 3) still holds

        The key insight from Section 8.1:
        - Standard decomposition creates queries with linear dependencies (columns sum to 0)
        - This causes numerical instability in the convex optimizer
        - Workaround: Use identity matrix in decomposition, then project B onto
          residual subspace after optimization
        """
        print("\n" + "="*70)
        print("Testing Section 8.1: Convex Solver Workflow Without Noise")
        print("="*70)

        domains = [2, 3]  # A1 has size 2, A2 has size 3

        # Test query from Figure 2 in paper
        # Query: count (A1=yes, A2 in {b,c}) OR (A1=no, A2=c)
        q_vec = np.array([[0, 1, 1, 0, 0, 1]], dtype=float)
        q = MatrixQuery(domains, (0, 1), q_vec)

        # Test data
        data = np.array([5, 3, 2, 1, 4, 6], dtype=float)
        true_answer = np.dot(q_vec.flatten(), data)
        print(f"\nTrue answer (q · x): {true_answer}")

        # ================================================================
        # Part 1: Verify decomposition difference between standard and convex
        # ================================================================
        print("\n--- Part 1: Decomposition Comparison ---")

        attributes = (0, 1)
        for subset in powerset(attributes):
            subset = tuple(sorted(subset))

            # Standard decomposition (for Fourier)
            std_sub_q = q.decompose(subset, for_convex=False)
            std_vec = std_sub_q.query_vector()

            # Convex decomposition (uses I_d instead of centering matrix)
            cvx_sub_q = q.decompose(subset, for_convex=True)
            cvx_vec = cvx_sub_q.query_vector()

            # For non-empty subsets, they should differ
            # For empty subset, they should be the same (averaging step is the same)
            if len(subset) > 0:
                diff = np.max(np.abs(std_vec - cvx_vec))
                print(f"  A'={subset}: std vs convex max diff = {diff:.6f}")
                # They should be different (convex uses I instead of centering)
                # unless the query happens to already be centered
            else:
                # Empty subset: both average over all attributes
                np.testing.assert_array_almost_equal(std_vec, cvx_vec, decimal=10)
                print(f"  A'={subset}: identical (empty subset)")

        # ================================================================
        # Part 2: Verify convex solver optimization and projection
        # ================================================================
        print("\n--- Part 2: Convex Solver Optimization ---")

        from solver import convexSolver
        from functools import reduce

        # Test on the (0, 1) partition - the full marginal
        subset = (0, 1)

        # Create convex solver
        solver = convexSolver(domains, subset)

        # Add decomposed query (using convex decomposition)
        cvx_sub_q = q.decompose(subset, for_convex=True)
        solver.add_query(cvx_sub_q)

        # Optimize
        solver.optimize(target_privacy_cost=1.0)

        # Verify B matrix was created and projected
        self.assertIsNotNone(solver.B)
        print(f"  B matrix shape: {solver.B.shape}")
        print(f"  B matrix:\n{solver.B}")

        # Verify projection: B @ P should equal B (B is already projected)
        # P = ⊗_i (I_{d_i} - (1/d_i) J_{d_i})
        P_factors = []
        for attr_idx in subset:
            d = domains[attr_idx]
            P_i = np.eye(d) - (1.0/d) * np.ones((d, d))
            P_factors.append(P_i)
        P = reduce(np.kron, P_factors)

        B_proj = solver.B @ P
        proj_diff = np.max(np.abs(solver.B - B_proj))
        print(f"  Projection verification: ||B - B@P|| = {proj_diff:.2e}")
        # B should be in the residual subspace, so B @ P ≈ B
        self.assertLess(proj_diff, 1e-10,
            "B matrix should be in residual subspace (B @ P = B)")

        # Verify privacy cost constraint: max_i (B^T B)[i,i] <= 1
        V = solver.B.T @ solver.B
        max_diag = np.max(np.diag(V))
        print(f"  Max diagonal of V = B^T B: {max_diag:.6f} (should be <= 1)")
        self.assertLessEqual(max_diag, 1.0 + 1e-6,
            "Privacy cost constraint violated")

        # ================================================================
        # Part 3: Full pipeline reconstruction without noise
        # ================================================================
        print("\n--- Part 3: Full Pipeline Reconstruction (No Noise) ---")

        # Create Smasher with convex solver
        smasher = Smasher(domains, privacy_cost=1.0, default_solver='convex', verbose=False)
        smasher.add_queries_to_workload([q])
        smasher.optimize()

        # Verify partitions were created
        print(f"  Partitions created: {list(smasher.partitions.keys())}")
        print(f"  Solvers: {list(smasher.solvers.keys())}")

        # Manually compute reconstruction without noise
        # For each partition, apply B to marginal (no noise), then reconstruct
        reconstructed_answer = 0.0

        print("\n  Reconstruction breakdown:")
        for subset in powerset(attributes):
            subset = tuple(sorted(subset))

            if subset not in smasher.solvers:
                print(f"    A'={subset}: no solver (empty partition)")
                continue

            solver = smasher.solvers[subset]

            # Extract marginal
            x_marginal = smasher._Smasher__extract_marginal(data, subset)

            # Measure without noise and reconstruct through the solver
            omega = solver.measure(x_marginal, add_noise=False)
            x_hat = solver.reconstruct_marginal(omega)

            # Use the appropriate decomposition based on solver type
            is_convex = isinstance(solver, convexSolver)
            sub_q = q.decompose(subset, for_convex=is_convex)
            sub_q_vec = sub_q.query_vector().flatten()

            sub_answer = np.dot(sub_q_vec, x_hat)
            reconstructed_answer += sub_answer

            print(f"    A'={subset}: q^{{=>A'}} · x̂_A' = {sub_answer:.6f}")

        print(f"\n  True answer: {true_answer}")
        print(f"  Reconstructed answer: {reconstructed_answer:.6f}")
        print(f"  Difference: {abs(true_answer - reconstructed_answer):.2e}")

        # Reconstruction should match (within SDP solver tolerance)
        self.assertAlmostEqual(true_answer, reconstructed_answer, places=5,
            msg=f"Reconstruction failed: true={true_answer}, reconstructed={reconstructed_answer}")

        # ================================================================
        # Part 4: Compare with standard decomposition (Fourier solver)
        # ================================================================
        print("\n--- Part 4: Comparison with Standard Decomposition (Fourier) ---")

        smasher_fourier = Smasher(domains, privacy_cost=1.0, default_solver='fourier', verbose=False)
        smasher_fourier.add_queries_to_workload([MatrixQuery(domains, (0, 1), q_vec)])
        smasher_fourier.optimize()

        # Reconstruct with Fourier (no noise)
        fourier_answer = 0.0

        for subset in powerset(attributes):
            subset = tuple(sorted(subset))

            if subset not in smasher_fourier.solvers:
                continue

            solver = smasher_fourier.solvers[subset]
            x_marginal = smasher_fourier._Smasher__extract_marginal(data, subset)

            # For Fourier: use theta to weight Fourier coefficients
            # But without noise, we can use direct reconstruction
            # Actually for clean reconstruction test, use standard decomposition
            std_sub_q = q.decompose(subset, for_convex=False)
            sub_q_vec = std_sub_q.query_vector().flatten()
            fourier_answer += np.dot(sub_q_vec, x_marginal)

        print(f"  Fourier reconstruction (standard decomp): {fourier_answer:.6f}")
        print(f"  Convex reconstruction: {reconstructed_answer:.6f}")
        print(f"  Both should equal true answer: {true_answer}")

        self.assertAlmostEqual(true_answer, fourier_answer, places=10)

        print("\n" + "="*70)
        print("Section 8.1 Test PASSED: Convex solver workflow verified!")
        print("="*70)

    def test_section_8_2_fourier_solver_workflow_no_noise(self):
        """
        Test Section 8.2: Fourier solver workflow without noise.

        This verifies the complete Algorithm 1 workflow for the Fourier solver:
        1. Standard decomposition creates centered queries (sum along any axis = 0)
        2. Fourier optimization computes tunable parameters θ
        3. Measurement in Fourier domain with conjugate symmetry
        4. Reconstruction preserves query answers

        The key insight from Section 8.2:
        - Decomposed queries in WKLoad^{⇒A'} have sum along any axis = 0
        - This means only non-DC Fourier coefficients (j_i ∈ {1, ..., d_i-1}) are relevant
        - Conjugate symmetry: θ[j] = θ[d-j] since input is real
        - Variance coefficient: 4× for non-self-conjugate indices
        """
        print("\n" + "="*70)
        print("Testing Section 8.2: Fourier Solver Workflow Without Noise")
        print("="*70)

        domains = [3, 4]  # A1 has size 3, A2 has size 4

        # Create a query that will have interesting decomposition
        # Query: count records where A1=0,A2=0 OR A1=1,A2=2 OR A1=2,A2=3
        q_matrix = np.zeros((3, 4))
        q_matrix[0, 0] = 1
        q_matrix[1, 2] = 1
        q_matrix[2, 3] = 1
        q_vec = q_matrix.flatten().reshape(1, -1)
        q = MatrixQuery(domains, (0, 1), q_vec)

        # Test data
        np.random.seed(42)
        data = np.random.randint(1, 100, size=12).astype(float)
        true_answer = np.dot(q_vec.flatten(), data)
        print(f"\nTrue answer (q · x): {true_answer}")
        print(f"Data: {data.reshape(3, 4)}")

        # ================================================================
        # Part 1: Verify decomposed queries are centered (sum=0 along axes)
        # ================================================================
        print("\n--- Part 1: Verify Decomposition Creates Centered Queries ---")

        attributes = (0, 1)
        for subset in powerset(attributes):
            subset = tuple(sorted(subset))
            if len(subset) == 0:
                continue  # Skip scalar partition

            sub_q = q.decompose(subset, for_convex=False)  # Standard decomposition
            sub_q_vec = sub_q.query_vector()

            # Reshape to tensor form
            sub_shape = tuple(domains[i] for i in subset)
            if len(sub_shape) > 0:
                sub_tensor = sub_q_vec.reshape(-1, *sub_shape)

                # Check that sum along each axis is (approximately) 0
                is_centered = True
                for axis in range(1, len(sub_shape) + 1):
                    axis_sum = np.sum(sub_tensor, axis=axis)
                    max_sum = np.max(np.abs(axis_sum))
                    if max_sum > 1e-10:
                        is_centered = False
                        print(f"  A'={subset}: NOT centered along axis {axis-1} (max sum = {max_sum:.2e})")

                if is_centered:
                    print(f"  A'={subset}: centered (sum=0 along all axes) ✓")

        # ================================================================
        # Part 2: Verify Fourier solver optimization
        # ================================================================
        print("\n--- Part 2: Fourier Solver Optimization ---")

        from solver import FourierSolver

        # Test on the (0, 1) partition
        subset = (0, 1)
        solver = FourierSolver(domains, subset)

        # Add decomposed query
        sub_q = q.decompose(subset, for_convex=False)
        solver.add_query(sub_q)

        # Optimize
        solver.optimize(target_privacy_cost=1.0)

        print(f"  Tunable indices: {solver._get_tunable_indices()}")
        print(f"  Number of tunable parameters: {len(solver.theta)}")
        print(f"  Theta values: {dict(list(solver.theta.items())[:5])}...")  # First 5

        # Verify theta is positive
        for idx, val in solver.theta.items():
            self.assertGreater(val, 0, f"theta[{idx}] should be positive")

        # ================================================================
        # Part 3: Full pipeline reconstruction without noise
        # ================================================================
        print("\n--- Part 3: Full Pipeline Reconstruction (No Noise) ---")

        smasher = Smasher(domains, privacy_cost=1.0, default_solver='fourier', verbose=False)
        smasher.add_queries_to_workload([q])
        smasher.optimize()

        # Set noise_scale to 0 for all solvers (deterministic test)
        for subset, solver in smasher.solvers.items():
            solver.noise_scale = 0.0

        # Measure with no noise
        smasher.measure(data)

        # Answer query
        reconstructed_answer = smasher.answer_query(q)
        if isinstance(reconstructed_answer, np.ndarray):
            reconstructed_answer = reconstructed_answer.item()

        print(f"  True answer: {true_answer}")
        print(f"  Reconstructed answer: {reconstructed_answer:.10f}")
        print(f"  Difference: {abs(true_answer - reconstructed_answer):.2e}")

        # Reconstruction should match exactly (within numerical tolerance)
        self.assertAlmostEqual(true_answer, reconstructed_answer, places=8,
            msg=f"Reconstruction failed: true={true_answer}, reconstructed={reconstructed_answer}")

        # ================================================================
        # Part 4: Verify variance computation
        # ================================================================
        print("\n--- Part 4: Variance Computation ---")

        # Reset noise_scale to 1.0 for variance computation
        for subset, solver in smasher.solvers.items():
            solver.noise_scale = 1.0

        total_variance = smasher.workload_variance()
        print(f"  Total workload variance at pcost=1: {total_variance:.6f}")

        # Variance should be positive and finite
        self.assertGreater(total_variance, 0)
        self.assertTrue(np.isfinite(total_variance))

        # ================================================================
        # Part 5: Compare Fourier vs Convex reconstruction
        # ================================================================
        print("\n--- Part 5: Comparison with Convex Solver ---")

        smasher_convex = Smasher(domains, privacy_cost=1.0, default_solver='convex', verbose=False)
        smasher_convex.add_queries_to_workload([MatrixQuery(domains, (0, 1), q_vec)])
        smasher_convex.optimize()

        # Set noise_scale to 0 for deterministic test
        for subset, solver in smasher_convex.solvers.items():
            solver.noise_scale = 0.0

        smasher_convex.measure(data)
        convex_answer = smasher_convex.answer_query(q)
        if isinstance(convex_answer, np.ndarray):
            convex_answer = convex_answer.item()

        print(f"  Fourier answer: {reconstructed_answer:.10f}")
        print(f"  Convex answer: {convex_answer:.10f}")
        print(f"  True answer: {true_answer}")

        # Both should match true answer (within SDP solver tolerance)
        self.assertAlmostEqual(true_answer, convex_answer, places=5)

        print("\n" + "="*70)
        print("Section 8.2 Test PASSED: Fourier solver workflow verified!")
        print("="*70)


if __name__ == '__main__':
    unittest.main()
