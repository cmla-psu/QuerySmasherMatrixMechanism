import unittest
import numpy as np
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from smasher import Smasher
from query import MatrixQuery

class TestSmasher(unittest.TestCase):
    def setUp(self):
        self.domains = [2]
        self.smasher = Smasher(self.domains, verbose=False)
        self.q = MatrixQuery([2], (0,), np.array([[1, 1]])) # Count all

    def test_workflow_steps(self):
        # 1. Add
        self.smasher.add_queries_to_workload([self.q])
        self.assertEqual(len(self.smasher.queries), 1)
        
        # 2. Decompose (called via optimize)
        self.smasher.optimize()
        
        # Should have partitions for subsets of (0,) -> (), (0,)
        # () is empty set partition
        self.assertIn((0,), self.smasher.partitions)
        self.assertIn((), self.smasher.partitions)
        
        self.assertTrue(self.smasher.is_optimized)
        
        # 3. Assembly verification
        # Check if solvers exist
        self.assertIn((0,), self.smasher.solvers)
        
    def test_empty_workload(self):
        s = Smasher([2])
        s.optimize()
        # Should not crash
        self.assertFalse(s.is_optimized) # optimize returns early if empty?

if __name__ == '__main__':
    unittest.main()
