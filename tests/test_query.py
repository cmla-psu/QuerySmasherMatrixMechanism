import unittest
import numpy as np
import torch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from query import MatrixQuery

class TestMatrixQuery(unittest.TestCase):
    def setUp(self):
        self.domains = [2, 3] # Dom sizes: 2, 3. Total 6.
        # Identity query on full domain
        self.q_vec = np.eye(6)
        self.query = MatrixQuery(self.domains, (0, 1), self.q_vec)

    def test_init(self):
        self.assertEqual(self.query.query_vector().shape, (6, 6))
        
    def test_tensor_view(self):
        # Shape should be (6, 2, 3) because 6 queries, marginal shape (2,3)
        # default batch_dim_last=True -> (2, 3, 6)
        t_view = self.query.tensor_view(add_batch_dim=True, batch_dim_last=True)
        self.assertEqual(t_view.shape, (2, 3, 6))
        
        # batch_dim_last=False -> (6, 2, 3)
        t_view_first = self.query.tensor_view(add_batch_dim=True, batch_dim_last=False)
        self.assertEqual(t_view_first.shape, (6, 2, 3))

    def test_decompose_marginal(self):
        # Decompose to attribute 0 (size 2)
        # Q matrix should project 6 -> 2
        # Logic: q_sub = q @ Q
        # For identity query, q_sub IS Q.
        
        sub_q = self.query.decompose((0,))
        sub_vec = sub_q.query_vector()
        
        # Expected Q for (0,): Q_0 x Q_1
        # 0 is in subset -> Q_0 is Centered P (2x2)
        # 1 not in subset -> Q_1 is Averaging (3x1)
        # Q = P x Avg. Shape (2*3, 2*1) = (6, 2).
        
        self.assertEqual(sub_vec.shape, (6, 2))
        
        # Verify values. Q_1 is column of 1/3.
        # Q_0 is [[2/3, -1/3]? No. I - J/d.
        # I_2 - [[0.5, 0.5],[0.5, 0.5]] = [[0.5, -0.5], [-0.5, 0.5]].
        # Wait, math in query.py:
        # Q = np.eye(d) - (1/d) * np.ones((d, d))
        
        P2 = np.array([[0.5, -0.5], [-0.5, 0.5]])
        Avg3 = np.array([[1/3], [1/3], [1/3]])
        
        Expected = np.kron(P2, Avg3)
        
        # diff
        diff = np.abs(sub_vec - Expected).max()
        self.assertLess(diff, 1e-9)

if __name__ == '__main__':
    unittest.main()
