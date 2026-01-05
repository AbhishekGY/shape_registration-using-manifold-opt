"""Cost functions and gradients for mesh registration.

This module implements the objective function for shape registration:
    L(R, t, d) = L_data(R, t, d) + lambda_smooth * L_smooth(d)

where:
- L_data measures alignment quality (point-to-point or point-to-plane)
- L_smooth penalizes non-smooth deformation fields
"""

import numpy as np
from scipy.spatial import KDTree
from typing import Tuple, Optional


class DataTerm:
    """Point-to-point distance cost for mesh alignment.

    Measures the sum of squared distances from transformed source vertices
    to their nearest neighbors on the target mesh.

    Cost: L_data = sum_i ||transformed_source_i - closest_target_i||^2
    """

    def __init__(self, target_vertices: np.ndarray):
        """Initialize with target mesh vertices.

        Args:
            target_vertices: (M, 3) array of target vertex positions
        """
        self.target = np.asarray(target_vertices, dtype=np.float64)
        if self.target.ndim != 2 or self.target.shape[1] != 3:
            raise ValueError(f"target_vertices must have shape (M, 3), got {self.target.shape}")

        # Build KD-tree for efficient nearest neighbor queries
        self.kdtree = KDTree(self.target)

        # Cache for last query results (useful when computing cost then gradient)
        self._last_query_points = None
        self._last_distances = None
        self._last_indices = None

    def compute_correspondences(
        self,
        transformed_source: np.ndarray,
        distance_threshold: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Find closest target point for each source point.

        Args:
            transformed_source: (N, 3) transformed source vertices
            distance_threshold: Optional max distance to consider valid correspondence.
                               Points beyond this get mask=False.

        Returns:
            closest_points: (N, 3) nearest target vertices
            indices: (N,) indices into target array
            mask: (N,) boolean array, True for valid correspondences
        """
        transformed_source = np.asarray(transformed_source)

        # Query KD-tree
        distances, indices = self.kdtree.query(transformed_source)

        # Cache results
        self._last_query_points = transformed_source.copy()
        self._last_distances = distances
        self._last_indices = indices

        closest_points = self.target[indices]

        # Create validity mask
        if distance_threshold is not None:
            mask = distances < distance_threshold
        else:
            mask = np.ones(len(transformed_source), dtype=bool)

        return closest_points, indices, mask

    def compute_cost(
        self,
        transformed_source: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> float:
        """Compute sum of squared distances to nearest points.

        Args:
            transformed_source: (N, 3) transformed source vertices
            weights: (N,) optional per-vertex weights

        Returns:
            cost: weighted sum of squared distances
        """
        closest_points, _, mask = self.compute_correspondences(transformed_source)

        # Compute squared distances
        diff = transformed_source - closest_points
        squared_distances = np.sum(diff ** 2, axis=1)

        if weights is not None:
            weights = np.asarray(weights)
            squared_distances = squared_distances * weights

        # Only count valid correspondences
        return np.sum(squared_distances[mask])

    def compute_gradient_wrt_transformed(
        self,
        transformed_source: np.ndarray,
        weights: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Compute gradient of cost w.r.t. transformed source vertices.

        The cost for vertex i is: ||p_i - q_i||^2 where q_i is closest target point.
        Gradient w.r.t. p_i is: 2 * (p_i - q_i)

        Args:
            transformed_source: (N, 3) transformed source vertices
            weights: (N,) optional per-vertex weights

        Returns:
            grad: (N, 3) gradient for each vertex
        """
        closest_points, _, mask = self.compute_correspondences(transformed_source)

        # Gradient is 2 * (transformed - closest)
        grad = 2.0 * (transformed_source - closest_points)

        if weights is not None:
            weights = np.asarray(weights)
            grad = grad * weights[:, np.newaxis]

        # Zero out gradient for invalid correspondences
        grad[~mask] = 0.0

        return grad

    def get_correspondence_distances(self) -> Optional[np.ndarray]:
        """Get distances from last correspondence query.

        Returns:
            (N,) distances or None if no query has been made
        """
        return self._last_distances


class PointToPlaneDataTerm:
    """Point-to-plane distance cost (more robust for surface alignment).

    Instead of measuring point-to-point distance, measures distance along
    the target surface normal. This is more robust to sliding motion along
    surfaces and converges faster for smooth surfaces.

    Cost: L = sum_i (n_i . (p_i - q_i))^2

    where n_i is the normal at closest target point q_i.
    """

    def __init__(self, target_vertices: np.ndarray, target_normals: np.ndarray):
        """Initialize with target vertices and normals.

        Args:
            target_vertices: (M, 3) target vertex positions
            target_normals: (M, 3) target vertex normals (should be unit length)
        """
        self.target = np.asarray(target_vertices, dtype=np.float64)
        self.normals = np.asarray(target_normals, dtype=np.float64)

        if self.target.shape != self.normals.shape:
            raise ValueError(
                f"target_vertices and target_normals must have same shape, "
                f"got {self.target.shape} and {self.normals.shape}"
            )

        self.kdtree = KDTree(self.target)

    def compute_cost(self, transformed_source: np.ndarray) -> float:
        """Compute point-to-plane cost.

        Args:
            transformed_source: (N, 3) transformed source vertices

        Returns:
            cost: sum of squared normal distances
        """
        distances, indices = self.kdtree.query(transformed_source)
        closest_points = self.target[indices]
        closest_normals = self.normals[indices]

        # Point-to-plane distance: dot product with normal
        diff = transformed_source - closest_points
        normal_distances = np.sum(diff * closest_normals, axis=1)

        return np.sum(normal_distances ** 2)

    def compute_gradient_wrt_transformed(self, transformed_source: np.ndarray) -> np.ndarray:
        """Compute gradient of point-to-plane cost.

        d/dp [(n . (p - q))^2] = 2 * (n . (p - q)) * n

        Args:
            transformed_source: (N, 3) transformed source vertices

        Returns:
            grad: (N, 3) gradient for each vertex
        """
        distances, indices = self.kdtree.query(transformed_source)
        closest_points = self.target[indices]
        closest_normals = self.normals[indices]

        diff = transformed_source - closest_points
        normal_distances = np.sum(diff * closest_normals, axis=1, keepdims=True)

        grad = 2.0 * normal_distances * closest_normals

        return grad


class SmoothnessRegularization:
    """Penalize differences in deformation between neighboring vertices.

    Encourages the deformation field to be smooth by penalizing differences
    in deformation vectors between connected vertices (edges).

    Cost: L_smooth = sum_{(i,j) in edges} ||d_i - d_j||^2

    This acts as a discrete Laplacian regularizer on the deformation field.
    """

    def __init__(self, edges: np.ndarray, weight: float = 1.0):
        """Initialize with mesh edge connectivity.

        Args:
            edges: (E, 2) array of vertex index pairs defining edges
            weight: Regularization strength (lambda_smooth)
        """
        self.edges = np.asarray(edges, dtype=np.int32)
        if self.edges.ndim != 2 or self.edges.shape[1] != 2:
            raise ValueError(f"edges must have shape (E, 2), got {self.edges.shape}")

        self.weight = float(weight)

        # Precompute sparse structure for efficient gradient computation
        self._build_adjacency_structure()

    def _build_adjacency_structure(self):
        """Build data structures for efficient gradient computation."""
        # Find max vertex index to determine size
        if len(self.edges) == 0:
            self._max_vertex = 0
            self._vertex_edges = {}
            return

        self._max_vertex = self.edges.max() + 1

        # For each vertex, store list of (edge_idx, is_first_vertex)
        # This allows efficient gradient accumulation
        self._vertex_edges = {i: [] for i in range(self._max_vertex)}
        for edge_idx, (i, j) in enumerate(self.edges):
            self._vertex_edges[i].append((edge_idx, True))   # vertex i is first
            self._vertex_edges[j].append((edge_idx, False))  # vertex j is second

    def compute_cost(self, d: np.ndarray) -> float:
        """Compute smoothness cost.

        Args:
            d: (N, 3) deformation field

        Returns:
            cost: lambda_smooth * sum of squared deformation differences
        """
        if len(self.edges) == 0:
            return 0.0

        d = np.asarray(d)

        # Get deformations at edge endpoints
        d_i = d[self.edges[:, 0]]  # (E, 3)
        d_j = d[self.edges[:, 1]]  # (E, 3)

        # Squared differences
        diff = d_i - d_j
        squared_norms = np.sum(diff ** 2, axis=1)

        return self.weight * np.sum(squared_norms)

    def compute_gradient(self, d: np.ndarray) -> np.ndarray:
        """Compute gradient of smoothness cost w.r.t. deformation.

        For edge (i, j) with cost ||d_i - d_j||^2:
            d/d(d_i) = 2 * (d_i - d_j)
            d/d(d_j) = -2 * (d_i - d_j)

        Args:
            d: (N, 3) deformation field

        Returns:
            grad: (N, 3) gradient for each vertex
        """
        d = np.asarray(d)
        N = len(d)
        grad = np.zeros_like(d)

        if len(self.edges) == 0:
            return grad

        # Vectorized computation of edge differences
        d_i = d[self.edges[:, 0]]  # (E, 3)
        d_j = d[self.edges[:, 1]]  # (E, 3)
        diff = d_i - d_j  # (E, 3)

        # Scale by 2 * weight
        scaled_diff = 2.0 * self.weight * diff

        # Accumulate gradients using np.add.at for efficiency
        # Gradient for vertex i: +2 * weight * (d_i - d_j) for each edge containing i
        # Gradient for vertex j: -2 * weight * (d_i - d_j) for each edge containing j
        np.add.at(grad, self.edges[:, 0], scaled_diff)
        np.add.at(grad, self.edges[:, 1], -scaled_diff)

        return grad


class RigidityRegularization:
    """Penalize non-rigid (non-affine) deformations.

    Encourages local deformations to be approximately rigid by penalizing
    deviations from local rotation. This is a simplified version of ARAP
    (As-Rigid-As-Possible) energy.

    For each vertex i and its neighbors N(i), penalizes:
        sum_{j in N(i)} ||d_j - d_i||^2 - ||v_j - v_i||^2

    This encourages the deformation to preserve local distances.
    """

    def __init__(
        self,
        edges: np.ndarray,
        original_vertices: np.ndarray,
        weight: float = 1.0
    ):
        """Initialize rigidity regularization.

        Args:
            edges: (E, 2) edge connectivity
            original_vertices: (N, 3) original vertex positions
            weight: Regularization strength
        """
        self.edges = np.asarray(edges, dtype=np.int32)
        self.vertices = np.asarray(original_vertices, dtype=np.float64)
        self.weight = float(weight)

        # Precompute original edge lengths
        if len(self.edges) > 0:
            v_i = self.vertices[self.edges[:, 0]]
            v_j = self.vertices[self.edges[:, 1]]
            self._original_lengths_sq = np.sum((v_j - v_i) ** 2, axis=1)
        else:
            self._original_lengths_sq = np.array([])

    def compute_cost(self, deformed_vertices: np.ndarray) -> float:
        """Compute rigidity cost.

        Args:
            deformed_vertices: (N, 3) deformed vertex positions

        Returns:
            cost: sum of squared edge length changes
        """
        if len(self.edges) == 0:
            return 0.0

        deformed = np.asarray(deformed_vertices)

        v_i = deformed[self.edges[:, 0]]
        v_j = deformed[self.edges[:, 1]]
        deformed_lengths_sq = np.sum((v_j - v_i) ** 2, axis=1)

        # Penalize change in squared edge length
        diff = deformed_lengths_sq - self._original_lengths_sq

        return self.weight * np.sum(diff ** 2)


def compute_rotation_gradient(
    source_vertices: np.ndarray,
    R: np.ndarray,
    grad_transformed: np.ndarray
) -> np.ndarray:
    """Compute gradient of cost w.r.t. rotation matrix R.

    Given that transformed[i] = R @ source[i] + d[i] + t, and we have
    grad_transformed = dL/d(transformed), we compute dL/dR using chain rule.

    Derivation:
        transformed_i = R @ source_i + d_i + t

        The (j, k) element of the gradient dL/dR is:
            sum_i (dL/d(transformed_i))_j * (d(transformed_i)_j / dR_jk)
            = sum_i grad_transformed[i, j] * source[i, k]

        In matrix form: dL/dR = grad_transformed.T @ source

    Args:
        source_vertices: (N, 3) original source vertex positions
        R: (3, 3) current rotation matrix (not used in computation but
           included for API consistency)
        grad_transformed: (N, 3) gradient w.r.t. transformed vertices

    Returns:
        G: (3, 3) Euclidean gradient w.r.t. R

    Note:
        This returns the Euclidean gradient. To use in manifold optimization,
        project onto tangent space using project_so3_tangent(R, G).
    """
    source_vertices = np.asarray(source_vertices)
    grad_transformed = np.asarray(grad_transformed)

    # G[j, k] = sum_i grad_transformed[i, j] * source[i, k]
    G = grad_transformed.T @ source_vertices  # (3, 3)

    return G


def compute_translation_gradient(grad_transformed: np.ndarray) -> np.ndarray:
    """Compute gradient of cost w.r.t. translation vector t.

    Since transformed[i] = R @ source[i] + d[i] + t, the gradient w.r.t. t
    is simply the sum of gradients w.r.t. each transformed vertex.

    Args:
        grad_transformed: (N, 3) gradient w.r.t. transformed vertices

    Returns:
        g_t: (3,) gradient w.r.t. translation
    """
    return np.sum(grad_transformed, axis=0)


def compute_deformation_gradient(
    grad_transformed: np.ndarray,
    smoothness_reg: Optional[SmoothnessRegularization] = None,
    d: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compute gradient of cost w.r.t. deformation field d.

    The total gradient combines:
    1. Data term: dL_data/d(d_i) = dL_data/d(transformed_i) (since d(transformed)/dd = I)
    2. Smoothness term: dL_smooth/d(d_i) from SmoothnessRegularization

    Args:
        grad_transformed: (N, 3) gradient of data term w.r.t. transformed vertices
        smoothness_reg: Optional smoothness regularizer
        d: (N, 3) current deformation field (needed if smoothness_reg is provided)

    Returns:
        g_d: (N, 3) total gradient w.r.t. deformation field
    """
    g_d = grad_transformed.copy()

    if smoothness_reg is not None and d is not None:
        g_d = g_d + smoothness_reg.compute_gradient(d)

    return g_d


def numerical_gradient_check(
    cost_fn,
    params: np.ndarray,
    analytical_grad: np.ndarray,
    eps: float = 1e-5,
    rtol: float = 1e-4,
    atol: float = 1e-6
) -> Tuple[bool, float, np.ndarray]:
    """Verify analytical gradient using finite differences.

    Computes numerical gradient and compares to provided analytical gradient.
    Useful for debugging gradient implementations.

    Args:
        cost_fn: Function that takes params and returns scalar cost
        params: Current parameter values (flattened)
        analytical_grad: Analytical gradient (same shape as params)
        eps: Finite difference step size
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        passed: True if gradients match within tolerance
        max_error: Maximum relative error
        numerical_grad: Computed numerical gradient
    """
    params = params.flatten()
    analytical_grad = analytical_grad.flatten()

    numerical_grad = np.zeros_like(params)

    for i in range(len(params)):
        params_plus = params.copy()
        params_minus = params.copy()
        params_plus[i] += eps
        params_minus[i] -= eps

        cost_plus = cost_fn(params_plus)
        cost_minus = cost_fn(params_minus)

        numerical_grad[i] = (cost_plus - cost_minus) / (2 * eps)

    # Compute relative error
    denom = np.maximum(np.abs(analytical_grad), np.abs(numerical_grad))
    denom = np.maximum(denom, 1e-8)  # Avoid division by zero

    rel_error = np.abs(analytical_grad - numerical_grad) / denom
    max_error = np.max(rel_error)

    passed = np.allclose(analytical_grad, numerical_grad, rtol=rtol, atol=atol)

    return passed, max_error, numerical_grad.reshape(analytical_grad.shape)
