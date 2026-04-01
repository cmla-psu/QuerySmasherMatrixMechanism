"""
Query classes for QuerySmasher.

Defines abstract Query interface and concrete implementations
(MatrixQuery, RangeQuery, etc.)
"""

from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union
import numpy as np
import torch
from functools import reduce


class Query(ABC):
    """
    Abstract base class representing a collection of queries.
    Subclasses represent different query types (matrix-based, range queries, etc.)
    """
    
    def __init__(self, domains: List[int], attributes: Tuple[int, ...], weight: float = 1.0):
        """
        Args:
            domains: List of domain sizes for all attributes
            attributes: Tuple of attribute indices used by this query
            weight: Weight for variance minimization
        """
        self.domains = domains
        self.attributes = attributes
        self.weight = weight
        self._cached_vector = None
        self._cached_tensor = None
        self._use_cache = True
    
    @abstractmethod
    def query_vector(self) -> np.ndarray:
        """
        Returns the query as a vector (flattened).
        Uses caching if enabled.
        
        Returns:
            Query matrix where each row is a query. Shape: (n_queries, domain_size)
        """
        pass
    
    @abstractmethod
    def tensor_view(self, add_batch_dim: bool = False, batch_dim_last: bool = True) -> torch.Tensor:
        """
        Returns the query as a tensor.
        
        Args:
            add_batch_dim: Whether to add a batch dimension
            batch_dim_last: If True, batch dimension is last; else first
        
        Returns:
            Tensor of shape (*marginal_shape, batch_size) or (batch_size, *marginal_shape)
        """
        pass
    
    def attributes_used(self) -> Tuple[int, ...]:
        """Returns tuple of attribute indices used by this query."""
        return self.attributes
    
    @abstractmethod
    def decompose(self, subset: Tuple[int, ...], for_convex: bool = False) -> 'Query':
        """
        Decomposes this query for a subset of attributes.
        Returns q^{=>subset} as defined in QuerySmasher.

        Args:
            subset: Subset of self.attributes to decompose onto
            for_convex: If True, use identity matrix for attributes in subset
                        (Section 7.1 workaround for convex solver numerical stability).
                        The convex solver projects onto residual subspace after optimization.

        Returns:
            A new Query object representing q^{=>subset}
        """
        pass
    
    def set_caching(self, use_cache: bool):
        """Enable or disable caching of query representations."""
        self._use_cache = use_cache
        if not use_cache:
            self._cached_vector = None
            self._cached_tensor = None
    
    def _compute_decomposition_matrix(self, subset: Tuple[int, ...],
                                        for_convex: bool = False) -> np.ndarray:
        """
        Helper method to compute Q matrix for decomposition.

        Returns Q = Q_{i1} ⊗ Q_{i2} ⊗ ... ⊗ Q_{il} from Equation 1

        Args:
            subset: Subset of attributes A' to decompose onto
            for_convex: If True, use identity matrix I_d for attributes in subset
                        instead of centering matrix (I_d - (1/d)*J_d).
                        This is the numerical stability workaround from Section 7.1.
                        The convex solver will project onto the residual subspace
                        after optimization.
        """
        Q_matrices = []
        # Iterate over attributes in the order they appear in self.attributes
        # This assumes self.attributes is sorted or ordered consistently with the domain list?
        # Typically attributes A is an ordered set.
        for attr in self.attributes:
            d = self.domains[attr]
            if attr in subset:
                if for_convex:
                    # Section 7.1 workaround: use I_d instead of centering matrix
                    # The convex solver will project onto residual subspace after optimization
                    Q = np.eye(d)
                else:
                    # Standard decomposition: Q = I_d - (1/d) * 1_d @ 1_d^T
                    # Shape: (d, d)
                    Q = np.eye(d) - (1/d) * np.ones((d, d))
            else:
                # Q = (1/d) * 1_d (column vector)
                # Shape: (d, 1)
                Q = (1/d) * np.ones((d, 1))
            Q_matrices.append(Q)

        # Compute Kronecker product of all Q matrices
        if not Q_matrices:
            return np.array([[1.0]])

        Q_total = reduce(np.kron, Q_matrices)
        return Q_total


class KronQuery(Query):
    """
    Query stored as Kronecker product of per-attribute factor matrices.

    For attributes (i_1, ..., i_l) with factors [F_1, ..., F_l]:
      - Full query matrix = kron(F_1, ..., F_l)
      - F_k has shape (n_queries_k, d_{i_k})

    This avoids materializing the full dense matrix, enabling large multi-attribute
    partitions that would be infeasible with MatrixQuery.
    """

    def __init__(self, domains: List[int], attributes: Tuple[int, ...],
                 factors: List[np.ndarray], weight: float = 1.0):
        """
        Args:
            domains: List of domain sizes for all attributes
            attributes: Tuple of attribute indices
            factors: List of np.ndarray, one per attribute in `attributes`.
                     factors[i] has shape (n_queries_i, domains[attributes[i]])
            weight: Weight for variance minimization
        """
        super().__init__(domains, attributes, weight)
        self.factors = [np.atleast_2d(np.array(f, dtype=float)) for f in factors]
        # Validate
        for i, attr in enumerate(attributes):
            if self.factors[i].shape[1] != domains[attr]:
                raise ValueError(
                    f"Factor {i} column dim {self.factors[i].shape[1]} != "
                    f"domain size {domains[attr]} for attribute {attr}")

    def query_vector(self) -> np.ndarray:
        """Materialize the full dense query matrix via Kronecker product."""
        return reduce(np.kron, self.factors)

    def tensor_view(self, add_batch_dim: bool = False, batch_dim_last: bool = True) -> torch.Tensor:
        """Returns the query as a tensor (materializes dense Kronecker product)."""
        matrix = self.query_vector()
        marginal_shape = tuple(self.domains[i] for i in self.attributes)
        n_queries = matrix.shape[0]
        tensor = torch.tensor(matrix).reshape(n_queries, *marginal_shape)
        if batch_dim_last:
            dims = list(range(1, len(marginal_shape) + 1)) + [0]
            tensor = tensor.permute(*dims)
        if not add_batch_dim and n_queries == 1:
            tensor = tensor.squeeze()
        return tensor

    def decompose(self, subset: Tuple[int, ...], for_convex: bool = False) -> 'Query':
        """
        Per-factor decomposition — same math as MatrixQuery but on small matrices.

        For each attribute i in self.attributes:
          - If i in subset: Q_i = I_d (for_convex=True) or I_d - (1/d)*J_d  → shape (d, d)
          - If i not in subset: Q_i = (1/d)*ones(d,1)  → shape (d, 1)

        New factor G_i = F_i @ Q_i.  The full decomposed query is kron(G_1,...,G_l).
        For non-subset attrs, G_i is (n_i, 1), so the Kron product has
        cols = prod(d_j for j in subset) as required.

        Returns KronQuery on subset if subset == self.attributes (all factors kept),
        otherwise MatrixQuery (since mixed factor shapes don't fit KronQuery).
        """
        new_factors = []
        subset_factors = []
        subset_attrs = []

        for attr, F in zip(self.attributes, self.factors):
            d = self.domains[attr]
            if attr in subset:
                if for_convex:
                    Q = np.eye(d)
                else:
                    Q = np.eye(d) - (1.0 / d) * np.ones((d, d))
                G = F @ Q  # (n_i, d)
                new_factors.append(G)
                subset_factors.append(G)
                subset_attrs.append(attr)
            else:
                G = F @ ((1.0 / d) * np.ones((d, 1)))  # (n_i, 1)
                new_factors.append(G)

        subset_sorted = tuple(sorted(subset))

        # If all attrs are in subset, return KronQuery (no collapsed factors)
        if len(subset_factors) == len(self.factors):
            return KronQuery(self.domains, subset_sorted, subset_factors, self.weight)

        if not subset_factors:
            # All attributes collapsed — scalar partition
            mat = reduce(np.kron, new_factors)  # (prod n_i, 1)
            return MatrixQuery(self.domains, tuple(), mat, self.weight)

        # Mixed case: some factors collapsed to (n_i, 1).
        # Fold collapsed factors into weight: each collapsed G_i is (n_i, 1),
        # and its contribution to the Kronecker product is a scalar multiplier
        # per combination of rows.  The total weight multiplier is:
        #   sum over all row combinations of collapsed factors of (product of values)^2
        # = prod_collapsed sum_{row} G_i[row,0]^2
        # This preserves the correct c spectral coefficients.
        weight_mult = 1.0
        for attr, F in zip(self.attributes, self.factors):
            if attr not in subset:
                d = self.domains[attr]
                collapsed = F @ ((1.0 / d) * np.ones((d, 1)))  # (n_i, 1)
                weight_mult *= np.sum(collapsed ** 2)
        return KronQuery(self.domains, subset_sorted, subset_factors,
                         self.weight * weight_mult)

    def gram_factors(self) -> List[np.ndarray]:
        """Returns [F_i^T @ F_i for each factor] — what the Kron convex solver needs."""
        return [F.T @ F for F in self.factors]

    def num_queries_per_factor(self) -> List[int]:
        """Returns number of query rows per factor."""
        return [F.shape[0] for F in self.factors]


class MatrixQuery(Query):
    """
    Concrete implementation of Query using an explicit matrix representation.
    Each row of the matrix corresponds to a linear query.
    """

    def __init__(self, domains: List[int], attributes: Tuple[int, ...], 
                 matrix: np.ndarray, weight: float = 1.0):
        """
        Args:
            domains: List of domain sizes
            attributes: Tuple of attribute indices
            matrix: Numpy array of shape (n_queries, product_of_domains_in_attributes)
            weight: Weight for variance minimization
        """
        super().__init__(domains, attributes, weight)
        self.matrix = np.array(matrix, dtype=float)
        
        # Validation
        expected_dim = 1
        for attr in attributes:
            expected_dim *= domains[attr]
            
        if self.matrix.ndim == 1:
            self.matrix = self.matrix.reshape(1, -1)
            
        if self.matrix.shape[1] != expected_dim:
            raise ValueError(f"Matrix column dimension {self.matrix.shape[1]} does not match "
                             f"expected domain size {expected_dim} for attributes {attributes}")

    def query_vector(self) -> np.ndarray:
        return self.matrix

    def tensor_view(self, add_batch_dim: bool = False, batch_dim_last: bool = True) -> torch.Tensor:
        # Calculate shape of the marginal tensor
        marginal_shape = tuple(self.domains[i] for i in self.attributes)
        n_queries = self.matrix.shape[0]
        
        # Reshape matrix rows to tensor
        # Matrix is (n_queries, flat_dim)
        # We need (n_queries, *marginal_shape)
        tensor = torch.tensor(self.matrix).reshape(n_queries, *marginal_shape)
        
        if batch_dim_last:
            # Move batch dimension (0) to last
            # Permute: (1, 2, ..., k, 0)
            dims = list(range(1, len(marginal_shape) + 1)) + [0]
            tensor = tensor.permute(*dims)
        
        if not add_batch_dim and n_queries == 1:
            tensor = tensor.squeeze()
            
        return tensor

    def decompose(self, subset: Tuple[int, ...], for_convex: bool = False) -> 'MatrixQuery':
        """
        Decompose this query for a subset of attributes.

        Args:
            subset: Subset of self.attributes to decompose onto
            for_convex: If True, use identity matrix for attributes in subset
                        (Section 7.1 workaround for convex solver).

        Returns:
            A new MatrixQuery representing q^{=>subset}
        """
        # 1. Compute projection matrix Q
        # Q has shape (original_dim, new_dim)
        # q_new = q_old @ Q
        # q_old shape: (n_queries, original_dim)
        # Q shape must be (original_dim, new_dim) logically for post-multiplication

        Q = self._compute_decomposition_matrix(subset, for_convex=for_convex)

        # 2. Compute new query matrix
        new_matrix = self.matrix @ Q

        # 3. Return new MatrixQuery
        return MatrixQuery(self.domains, subset, new_matrix, self.weight)
