"""Core data structures for 3D mesh registration."""

import numpy as np
from typing import Optional, Tuple


class Mesh:
    """Represents a 3D triangular mesh."""

    def __init__(self, vertices: np.ndarray, faces: np.ndarray):
        """
        Args:
            vertices: (N, 3) array of vertex positions
            faces: (M, 3) array of vertex indices forming triangles
        """
        self.vertices = vertices.astype(np.float64)
        self.faces = faces.astype(np.int32)
        self._edges = None  # Lazy computation

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def center_of_mass(self) -> np.ndarray:
        """Compute center of mass.

        Returns:
            (3,) array: center position
        """
        return np.mean(self.vertices, axis=0)

    def bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Compute axis-aligned bounding box.

        Returns:
            min_corner: (3,) array
            max_corner: (3,) array
        """
        return np.min(self.vertices, axis=0), np.max(self.vertices, axis=0)

    def normalize(self) -> 'Mesh':
        """Center mesh at origin and scale to unit bounding box.

        Returns:
            New normalized mesh
        """
        centered = self.vertices - self.center_of_mass()
        min_c, max_c = np.min(centered, axis=0), np.max(centered, axis=0)
        scale = np.max(max_c - min_c)
        normalized = centered / scale if scale > 0 else centered
        return Mesh(normalized, self.faces)

    def get_edges(self) -> np.ndarray:
        """Extract unique edges from faces.

        Returns:
            (E, 2) array of vertex index pairs
        """
        if self._edges is not None:
            return self._edges

        edges_set = set()
        for face in self.faces:
            for i in range(3):
                v1, v2 = face[i], face[(i + 1) % 3]
                edge = tuple(sorted([v1, v2]))
                edges_set.add(edge)

        self._edges = np.array(list(edges_set), dtype=np.int32)
        return self._edges

    def copy(self) -> 'Mesh':
        """Create deep copy."""
        return Mesh(self.vertices.copy(), self.faces.copy())


class RegistrationState:
    """Represents current state of registration optimization."""

    def __init__(
        self,
        R: np.ndarray,
        t: np.ndarray,
        d: np.ndarray,
        cost: float = np.inf
    ):
        """
        Args:
            R: (3, 3) rotation matrix in SO(3)
            t: (3,) translation vector
            d: (N, 3) per-vertex deformation displacements
            cost: Current cost value
        """
        self.R = R.copy()
        self.t = t.copy()
        self.d = d.copy()
        self.cost = cost

    def transform_vertices(self, vertices: np.ndarray) -> np.ndarray:
        """Apply transformation to vertices.

        Args:
            vertices: (N, 3) array

        Returns:
            (N, 3) transformed vertices: R @ v_i + d_i + t for each i
        """
        # Rotation: (N, 3) @ (3, 3).T = (N, 3)
        rotated = vertices @ self.R.T
        return rotated + self.d + self.t

    def copy(self) -> 'RegistrationState':
        """Create deep copy."""
        return RegistrationState(self.R, self.t, self.d, self.cost)
