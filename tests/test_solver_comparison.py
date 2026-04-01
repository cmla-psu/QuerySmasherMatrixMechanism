"""
Tests comparing Fourier and Convex solvers.

Verifies that both solvers produce identical noiseless answers on:
1. Simple queries (2x3 domain)
2. Demographic-style queries (age × race, like Hispanic/racial/age groups)
3. Circular range queries
4. All range queries with emphasis weights (from README)

Key insight: Without noise, both solvers must give the exact same answer
because Theorem 3 guarantees reconstruction regardless of decomposition method.
With noise, they have different variance but should match on circular workloads (Section 7.2).
"""

import unittest
import numpy as np
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smasher import Smasher, powerset
from query import MatrixQuery
from solver import convexSolver, FourierSolver


def build_range_query_1d(domain_size, lo, hi, circular=False):
    """
    Build a 1D range query vector [lo, hi) on a domain of given size.
    If circular=True, wraps around (e.g., lo=8, hi=2 on domain 10 → {8,9,0,1}).
    """
    v = np.zeros(domain_size)
    if circular and lo > hi:
        # Wrap: [lo, domain_size) ∪ [0, hi)
        for i in range(lo, domain_size):
            v[i] = 1.0
        for i in range(0, hi):
            v[i] = 1.0
    else:
        for i in range(lo, hi):
            v[i] = 1.0
    return v


def build_all_range_queries_1d(domain_size, circular=False):
    """Build all contiguous range queries on a 1D domain."""
    queries = []
    for lo in range(domain_size):
        for length in range(1, domain_size + 1):
            hi = lo + length
            if circular:
                hi = hi % domain_size if hi > domain_size else hi
                v = build_range_query_1d(domain_size, lo, hi, circular=True)
            else:
                if hi > domain_size:
                    continue
                v = build_range_query_1d(domain_size, lo, hi, circular=False)
            queries.append(v)
    return np.array(queries)


class TestFourierVsConvexNoiseless(unittest.TestCase):
    """
    Test that Fourier and Convex solvers give identical answers without noise.

    Both solvers implement Theorem 3 reconstruction, so noiseless answers must
    match the true answer regardless of which solver is used.
    """

    def _run_comparison(self, domains, attributes, q_matrix, data, test_name,
                        places=4):
        """
        Run both Fourier and Convex pipelines without noise and compare answers.
        """
        q_fourier = MatrixQuery(domains, attributes, q_matrix.copy())
        q_convex = MatrixQuery(domains, attributes, q_matrix.copy())

        # True answer
        marginal_size = 1
        for a in attributes:
            marginal_size *= domains[a]
        # If query is on full domain, data is the marginal
        # If query is on a subset, we need to extract the marginal
        # For simplicity, we use the full Smasher pipeline

        true_answer = q_matrix @ data

        # --- Fourier pipeline (noise=False) ---
        smasher_f = Smasher(domains, privacy_cost=1.0, default_solver='fourier',
                            noise=False, verbose=False)
        smasher_f.add_queries_to_workload([q_fourier])
        smasher_f.optimize()
        smasher_f.measure(data)
        ans_f = smasher_f.answer_query(q_fourier)
        if isinstance(ans_f, np.ndarray):
            ans_f = ans_f.flatten()

        # --- Convex pipeline (noise=False) ---
        smasher_c = Smasher(domains, privacy_cost=1.0, default_solver='convex',
                            noise=False, verbose=False)
        smasher_c.add_queries_to_workload([q_convex])
        smasher_c.optimize()
        smasher_c.measure(data)
        ans_c = smasher_c.answer_query(q_convex)
        if isinstance(ans_c, np.ndarray):
            ans_c = ans_c.flatten()

        # Both should match true answer
        np.testing.assert_array_almost_equal(
            ans_f, true_answer, decimal=places,
            err_msg=f"[{test_name}] Fourier answer != true answer"
        )
        np.testing.assert_array_almost_equal(
            ans_c, true_answer, decimal=places,
            err_msg=f"[{test_name}] Convex answer != true answer"
        )
        np.testing.assert_array_almost_equal(
            ans_f, ans_c, decimal=places,
            err_msg=f"[{test_name}] Fourier != Convex"
        )

        return ans_f, ans_c, true_answer

    # ================================================================
    # 1. Simple queries (from existing tests)
    # ================================================================

    def test_simple_2x3(self):
        """Simple counting query on 2×3 domain."""
        domains = [2, 3]
        q_vec = np.array([[0, 1, 1, 0, 0, 1]], dtype=float)
        data = np.array([5, 3, 2, 1, 4, 6], dtype=float)
        self._run_comparison(domains, (0, 1), q_vec, data, "simple_2x3")

    def test_identity_2x3(self):
        """Identity query (all 6 cells) on 2×3 domain."""
        domains = [2, 3]
        q_mat = np.eye(6)
        data = np.array([5, 3, 2, 1, 4, 6], dtype=float)
        self._run_comparison(domains, (0, 1), q_mat, data, "identity_2x3")

    # ================================================================
    # 2. Demographic queries: Age × Race
    #    README: "Hispanic, white, under over 18 non hispanic/Asian /
    #             over under 18 / how many racial pop, 18-65, 0-17, 65+"
    # ================================================================

    def test_demographic_age_race(self):
        """
        Age groups × Race categories.

        Domain:
          Age (attr 0): 3 categories [0-17, 18-65, 65+]
          Race (attr 1): 4 categories [Hispanic, White, Asian, Other]

        Queries:
          - Count Hispanic population
          - Count under-18 population
          - Count Hispanic under-18
          - Count non-Hispanic over-18 (combined: 18-65 + 65+, non-Hispanic)
          - Count total population
        """
        domains = [3, 4]  # Age: 3, Race: 4
        # Marginal is 3×4 = 12 cells
        # Layout (row-major): age=0-17 × [Hisp, White, Asian, Other],
        #                      age=18-65 × [...], age=65+ × [...]

        # Synthetic census-like data
        data = np.array([
            # 0-17:  Hisp  White  Asian  Other
                     120,   200,   80,    50,
            # 18-65: Hisp  White  Asian  Other
                     300,   500,  150,   100,
            # 65+:   Hisp  White  Asian  Other
                      80,   250,   40,    30,
        ], dtype=float)

        attributes = (0, 1)

        # Query 1: Count all Hispanic (race=0, any age)
        q_hispanic = np.zeros((1, 12))
        q_hispanic[0, 0] = 1   # 0-17, Hispanic
        q_hispanic[0, 4] = 1   # 18-65, Hispanic
        q_hispanic[0, 8] = 1   # 65+, Hispanic

        # Query 2: Count all under-18 (age=0, any race)
        q_under18 = np.zeros((1, 12))
        q_under18[0, 0:4] = 1  # 0-17, all races

        # Query 3: Count Hispanic under-18
        q_hisp_under18 = np.zeros((1, 12))
        q_hisp_under18[0, 0] = 1  # 0-17, Hispanic

        # Query 4: Count non-Hispanic over-18 (ages 18-65 and 65+, races 1-3)
        q_nonhisp_over18 = np.zeros((1, 12))
        q_nonhisp_over18[0, 5:8] = 1   # 18-65, White/Asian/Other
        q_nonhisp_over18[0, 9:12] = 1  # 65+, White/Asian/Other

        # Query 5: Total population
        q_total = np.ones((1, 12))

        # Combine all into one workload
        q_all = np.vstack([q_hispanic, q_under18, q_hisp_under18,
                           q_nonhisp_over18, q_total])

        self._run_comparison(domains, attributes, q_all, data,
                             "demographic_age_race")

    def test_demographic_age_ranges(self):
        """
        Age range queries: 0-17, 18-65, 65+ on a finer-grained age domain.

        Domain:
          Age (attr 0): 10 buckets (decades: 0-9, 10-19, ..., 90-99)
          Sex (attr 1): 2 categories [Male, Female]

        Queries map coarse age groups to fine buckets:
          0-17  → buckets 0, 1
          18-65 → buckets 2, 3, 4, 5, 6
          65+   → buckets 7, 8, 9
        """
        domains = [10, 2]  # Age: 10 decades, Sex: 2
        data = np.random.RandomState(42).randint(10, 200, size=20).astype(float)
        attributes = (0, 1)

        # Query: count males in 0-17 (buckets 0-1, sex=0)
        q_male_017 = np.zeros((1, 20))
        q_male_017[0, 0] = 1   # age=0, male
        q_male_017[0, 2] = 1   # age=1, male

        # Query: count all people in 18-65 (buckets 2-6, both sexes)
        q_1865 = np.zeros((1, 20))
        for age_bucket in range(2, 7):
            q_1865[0, age_bucket * 2] = 1      # male
            q_1865[0, age_bucket * 2 + 1] = 1  # female

        # Query: count females 65+ (buckets 7-9, sex=1)
        q_female_65 = np.zeros((1, 20))
        for age_bucket in range(7, 10):
            q_female_65[0, age_bucket * 2 + 1] = 1  # female

        q_all = np.vstack([q_male_017, q_1865, q_female_65])
        self._run_comparison(domains, attributes, q_all, data,
                             "demographic_age_ranges")

    # ================================================================
    # 3. Circular range queries
    #    README: "circular with fourier, vs convex optimizer, those should be same"
    # ================================================================

    def test_circular_range_1d(self):
        """
        All circular range queries on a single attribute of size 5.

        Fourier solver is optimal for circular-shift invariant workloads.
        Both solvers should give identical noiseless answers.
        """
        d = 5
        domains = [d]
        data = np.array([10, 20, 30, 40, 50], dtype=float)
        attributes = (0,)

        q_mat = build_all_range_queries_1d(d, circular=True)
        self._run_comparison(domains, attributes, q_mat, data,
                             "circular_range_1d_d5")

    def test_circular_range_larger(self):
        """Circular range queries on domain size 8."""
        d = 8
        domains = [d]
        data = np.random.RandomState(123).randint(1, 100, size=d).astype(float)
        attributes = (0,)

        q_mat = build_all_range_queries_1d(d, circular=True)
        self._run_comparison(domains, attributes, q_mat, data,
                             "circular_range_1d_d8")

    def test_circular_range_2d(self):
        """
        Circular range queries on a 2D domain (4×5).

        Build all range queries on each dimension and take Kronecker products
        to form 2D range queries.
        """
        domains = [4, 5]
        data = np.random.RandomState(99).randint(1, 50, size=20).astype(float)
        attributes = (0, 1)

        # All circular range queries on dim 0
        q0 = build_all_range_queries_1d(4, circular=True)
        # All circular range queries on dim 1
        q1 = build_all_range_queries_1d(5, circular=True)

        # Build 2D queries as Kronecker products of a subset
        # (full Kronecker product would be large, use first 4 from each)
        rows = []
        for i in range(min(4, len(q0))):
            for j in range(min(5, len(q1))):
                rows.append(np.kron(q0[i], q1[j]))
        q_mat = np.array(rows)

        self._run_comparison(domains, attributes, q_mat, data,
                             "circular_range_2d_4x5")

    # ================================================================
    # 4. All range queries with emphasis (from README)
    #    "all range queries, extra emphasis, size 5 0-100 95, for weights"
    # ================================================================

    def test_all_range_queries_noncircular(self):
        """All non-circular range queries on domain size 5."""
        d = 5
        domains = [d]
        data = np.array([15, 25, 35, 45, 55], dtype=float)
        attributes = (0,)

        q_mat = build_all_range_queries_1d(d, circular=False)
        self._run_comparison(domains, attributes, q_mat, data,
                             "all_range_noncircular_d5")

    def test_weighted_range_queries(self):
        """
        Range queries with emphasis weights.

        Some range queries (like small ranges) get higher weight,
        simulating the "extra emphasis" from README.
        """
        d = 5
        domains = [d]
        data = np.array([10, 20, 30, 40, 50], dtype=float)
        attributes = (0,)

        # Build range queries with different weights
        # Small ranges get weight 5 (extra emphasis), larger get weight 1
        queries_f = []
        queries_c = []
        for lo in range(d):
            for length in range(1, d + 1):
                if lo + length > d:
                    continue
                v = build_range_query_1d(d, lo, lo + length, circular=False)
                w = 5.0 if length <= 2 else 1.0
                queries_f.append(MatrixQuery(domains, attributes,
                                             v.reshape(1, -1), weight=w))
                queries_c.append(MatrixQuery(domains, attributes,
                                             v.reshape(1, -1), weight=w))

        # Fourier pipeline
        smasher_f = Smasher(domains, privacy_cost=1.0, default_solver='fourier',
                            noise=False, verbose=False)
        smasher_f.add_queries_to_workload(queries_f)
        smasher_f.optimize()
        smasher_f.measure(data)

        # Convex pipeline
        smasher_c = Smasher(domains, privacy_cost=1.0, default_solver='convex',
                            noise=False, verbose=False)
        smasher_c.add_queries_to_workload(queries_c)
        smasher_c.optimize()
        smasher_c.measure(data)

        # Answer each query
        for i, (qf, qc) in enumerate(zip(queries_f, queries_c)):
            true_ans = qf.query_vector() @ data
            ans_f = smasher_f.answer_query(qf)
            ans_c = smasher_c.answer_query(qc)

            if isinstance(ans_f, np.ndarray):
                ans_f = ans_f.flatten()
            if isinstance(ans_c, np.ndarray):
                ans_c = ans_c.flatten()

            np.testing.assert_array_almost_equal(
                ans_f, true_ans, decimal=5,
                err_msg=f"Weighted query {i}: Fourier != true"
            )
            np.testing.assert_array_almost_equal(
                ans_c, true_ans, decimal=5,
                err_msg=f"Weighted query {i}: Convex != true"
            )

    # ================================================================
    # 5. 3-attribute queries (harder: more partitions)
    # ================================================================

    def test_three_attributes(self):
        """
        3-attribute query on Age × Race × Sex.

        Domains: Age=3, Race=2, Sex=2 → 12 cells
        This creates 2^3 = 8 partitions, testing the full decomposition.
        """
        domains = [3, 2, 2]
        data = np.random.RandomState(77).randint(5, 100, size=12).astype(float)
        attributes = (0, 1, 2)

        # Query: count all females (sex=1) regardless of age/race
        q_female = np.zeros((1, 12))
        # Layout: age × race × sex (row-major)
        # Index: age*4 + race*2 + sex
        for age in range(3):
            for race in range(2):
                q_female[0, age * 4 + race * 2 + 1] = 1  # sex=1

        # Query: count young (age=0) Hispanic (race=0) males (sex=0)
        q_specific = np.zeros((1, 12))
        q_specific[0, 0] = 1  # age=0, race=0, sex=0

        # Query: identity (all cells)
        q_id = np.eye(12)

        q_all = np.vstack([q_female, q_specific, q_id])
        self._run_comparison(domains, attributes, q_all, data,
                             "three_attributes_3x2x2")


class TestCircularVarianceComparison(unittest.TestCase):
    """
    Test that Fourier solver achieves lower or equal variance compared to
    Convex solver on circular-shift invariant workloads.

    From Section 7.2: Fourier is analytically optimal for circular workloads.
    The convex solver is a general-purpose SDP and may not achieve the same
    optimality, so we only check L_fourier <= L_convex.
    """

    def test_fourier_optimal_for_circular_d5(self):
        """Fourier should have lower variance than convex on circular queries (d=5)."""
        d = 5
        domains = [d]
        attributes = (0,)

        q_mat = build_all_range_queries_1d(d, circular=True)

        # --- Fourier solver directly ---
        solver_f = FourierSolver(domains, attributes)
        for i in range(q_mat.shape[0]):
            q = MatrixQuery(domains, attributes, q_mat[i:i+1])
            sub_q = q.decompose(attributes, for_convex=False)
            solver_f.add_query(sub_q)
        solver_f.optimize(target_privacy_cost=1.0)

        L_fourier = sum(q.weight * solver_f.query_variance(q)
                        for q in solver_f.queries)

        # --- Convex solver directly ---
        solver_c = convexSolver(domains, attributes)
        for i in range(q_mat.shape[0]):
            q = MatrixQuery(domains, attributes, q_mat[i:i+1])
            sub_q = q.decompose(attributes, for_convex=True)
            solver_c.add_query(sub_q)
        solver_c.optimize(target_privacy_cost=1.0)

        L_convex = sum(q.weight * solver_c.query_variance(q)
                       for q in solver_c.queries)

        print(f"\nCircular range queries on d={d}:")
        print(f"  L_fourier = {L_fourier:.6f}")
        print(f"  L_convex  = {L_convex:.6f}")
        print(f"  Ratio L_convex/L_fourier = {L_convex/L_fourier:.4f}")

        # Fourier should be at least as good (lower or equal variance)
        self.assertLessEqual(L_fourier, L_convex * 1.01,
            "Fourier should be optimal for circular workloads")

    def test_fourier_optimal_for_circular_d8(self):
        """Fourier should have lower variance than convex on circular queries (d=8)."""
        d = 8
        domains = [d]
        attributes = (0,)

        q_mat = build_all_range_queries_1d(d, circular=True)

        solver_f = FourierSolver(domains, attributes)
        for i in range(q_mat.shape[0]):
            q = MatrixQuery(domains, attributes, q_mat[i:i+1])
            sub_q = q.decompose(attributes, for_convex=False)
            solver_f.add_query(sub_q)
        solver_f.optimize(target_privacy_cost=1.0)

        L_fourier = sum(q.weight * solver_f.query_variance(q)
                        for q in solver_f.queries)

        solver_c = convexSolver(domains, attributes)
        for i in range(q_mat.shape[0]):
            q = MatrixQuery(domains, attributes, q_mat[i:i+1])
            sub_q = q.decompose(attributes, for_convex=True)
            solver_c.add_query(sub_q)
        solver_c.optimize(target_privacy_cost=1.0)

        L_convex = sum(q.weight * solver_c.query_variance(q)
                       for q in solver_c.queries)

        print(f"\nCircular range queries on d={d}:")
        print(f"  L_fourier = {L_fourier:.6f}")
        print(f"  L_convex  = {L_convex:.6f}")
        print(f"  Ratio L_convex/L_fourier = {L_convex/L_fourier:.4f}")

        self.assertLessEqual(L_fourier, L_convex * 1.01,
            "Fourier should be optimal for circular workloads")


if __name__ == '__main__':
    unittest.main()
