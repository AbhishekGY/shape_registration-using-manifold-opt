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

        Raises:
            ValueError: If vertices/faces have invalid shapes or face indices are out of bounds
        """
        vertices = np.asarray(vertices, dtype=np.float64)
        faces = np.asarray(faces, dtype=np.int32)

        # Validate shapes
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"vertices must have shape (N, 3), got {vertices.shape}")
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError(f"faces must have shape (M, 3), got {faces.shape}")

        # Validate face indices are within bounds
        if len(faces) > 0:
            if np.any(faces < 0) or np.any(faces >= len(vertices)):
                raise ValueError(
                    f"Face indices must be in range [0, {len(vertices)-1}], "
                    f"got range [{faces.min()}, {faces.max()}]"
                )

        self.vertices = vertices
        self.faces = faces

        # Lazy-computed cached properties
        self._edges = None
        self._face_normals = None
        self._vertex_normals = None

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    @property
    def n_edges(self) -> int:
        return len(self.get_edges())

    def center_of_mass(self) -> np.ndarray:
        """Compute center of mass (centroid of vertices).

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

    def bounding_box_diagonal(self) -> float:
        """Compute diagonal length of bounding box.

        Useful for scale-invariant distance thresholds.

        Returns:
            Diagonal length
        """
        min_c, max_c = self.bounding_box()
        return np.linalg.norm(max_c - min_c)

    def normalize(self) -> 'Mesh':
        """Center mesh at origin and scale to unit bounding box.

        Returns:
            New normalized mesh (does not modify original)
        """
        centered = self.vertices - self.center_of_mass()
        min_c, max_c = np.min(centered, axis=0), np.max(centered, axis=0)
        scale = np.max(max_c - min_c)
        if scale > 1e-10:
            normalized = centered / scale
        else:
            normalized = centered
        return Mesh(normalized, self.faces.copy())

    def get_edges(self) -> np.ndarray:
        """Extract unique edges from faces using vectorized operations.

        Returns:
            (E, 2) array of vertex index pairs, sorted so that edge[0] < edge[1]
        """
        if self._edges is not None:
            return self._edges

        if len(self.faces) == 0:
            self._edges = np.zeros((0, 2), dtype=np.int32)
            return self._edges

        # Extract all three edges from each face: (0,1), (1,2), (2,0)
        # Shape: (M, 3, 2) where M is number of faces
        edges = np.stack([
            self.faces[:, [0, 1]],
            self.faces[:, [1, 2]],
            self.faces[:, [2, 0]]
        ], axis=1)

        # Reshape to (3M, 2)
        edges = edges.reshape(-1, 2)

        # Sort each edge so smaller index comes first (for deduplication)
        edges = np.sort(edges, axis=1)

        # Remove duplicates by converting to structured array and using np.unique
        edges_structured = edges.view(dtype=[('v0', np.int32), ('v1', np.int32)])
        unique_edges = np.unique(edges_structured)

        # Convert back to regular array
        self._edges = unique_edges.view(np.int32).reshape(-1, 2)
        return self._edges

    def compute_face_normals(self) -> np.ndarray:
        """Compute unit normal vector for each face.

        Returns:
            (M, 3) array of face normals
        """
        if self._face_normals is not None:
            return self._face_normals

        # Get vertices of each face
        v0 = self.vertices[self.faces[:, 0]]  # (M, 3)
        v1 = self.vertices[self.faces[:, 1]]  # (M, 3)
        v2 = self.vertices[self.faces[:, 2]]  # (M, 3)

        # Compute edge vectors
        e1 = v1 - v0
        e2 = v2 - v0

        # Cross product gives normal (not normalized)
        normals = np.cross(e1, e2)

        # Normalize to unit length
        lengths = np.linalg.norm(normals, axis=1, keepdims=True)
        # Avoid division by zero for degenerate faces
        lengths = np.maximum(lengths, 1e-10)
        self._face_normals = normals / lengths

        return self._face_normals

    def compute_vertex_normals(self) -> np.ndarray:
        """Compute vertex normals by averaging adjacent face normals.

        Each vertex normal is the normalized sum of normals of all faces
        that contain that vertex.

        Returns:
            (N, 3) array of vertex normals
        """
        if self._vertex_normals is not None:
            return self._vertex_normals

        face_normals = self.compute_face_normals()

        # Accumulate face normals to each vertex
        vertex_normals = np.zeros_like(self.vertices)

        # Add face normal to each vertex of the face
        np.add.at(vertex_normals, self.faces[:, 0], face_normals)
        np.add.at(vertex_normals, self.faces[:, 1], face_normals)
        np.add.at(vertex_normals, self.faces[:, 2], face_normals)

        # Normalize
        lengths = np.linalg.norm(vertex_normals, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-10)
        self._vertex_normals = vertex_normals / lengths

        return self._vertex_normals

    def apply_transform(self, R: np.ndarray, t: np.ndarray) -> 'Mesh':
        """Apply rigid transformation to mesh.

        Args:
            R: (3, 3) rotation matrix
            t: (3,) translation vector

        Returns:
            New transformed mesh
        """
        transformed_vertices = (self.vertices @ R.T) + t
        return Mesh(transformed_vertices, self.faces.copy())

    def copy(self) -> 'Mesh':
        """Create deep copy."""
        return Mesh(self.vertices.copy(), self.faces.copy())

    def __repr__(self) -> str:
        return f"Mesh(n_vertices={self.n_vertices}, n_faces={self.n_faces})"


class RegistrationState:
    """Represents current state of registration optimization.

    The transformation model is: transformed_vertex[i] = R @ vertex[i] + d[i] + t
    where:
        - R is a global rotation (3x3 matrix in SO(3))
        - t is a global translation (3D vector)
        - d[i] is a per-vertex deformation (allows non-rigid alignment)
    """

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

        Raises:
            ValueError: If shapes are invalid or R is not a valid rotation
        """
        R = np.asarray(R, dtype=np.float64)
        t = np.asarray(t, dtype=np.float64)
        d = np.asarray(d, dtype=np.float64)

        # Validate shapes
        if R.shape != (3, 3):
            raise ValueError(f"R must have shape (3, 3), got {R.shape}")
        if t.shape != (3,):
            raise ValueError(f"t must have shape (3,), got {t.shape}")
        if d.ndim != 2 or d.shape[1] != 3:
            raise ValueError(f"d must have shape (N, 3), got {d.shape}")

        # Validate R is a rotation matrix (orthogonal with det = +1)
        if not self._is_valid_rotation(R):
            raise ValueError(
                "R must be a valid rotation matrix (R.T @ R = I, det(R) = 1). "
                f"Got det(R) = {np.linalg.det(R):.6f}, "
                f"||R.T @ R - I|| = {np.linalg.norm(R.T @ R - np.eye(3)):.6f}"
            )

        self.R = R.copy()
        self.t = t.copy()
        self.d = d.copy()
        self.cost = float(cost)

    @staticmethod
    def _is_valid_rotation(R: np.ndarray, tol: float = 1e-6) -> bool:
        """Check if matrix is a valid rotation matrix."""
        # Check orthogonality: R.T @ R should be identity
        if not np.allclose(R.T @ R, np.eye(3), atol=tol):
            return False
        # Check determinant: should be +1 (not -1, which would be reflection)
        if not np.isclose(np.linalg.det(R), 1.0, atol=tol):
            return False
        return True

    @classmethod
    def identity(cls, n_vertices: int) -> 'RegistrationState':
        """Create identity transformation state.

        Args:
            n_vertices: Number of vertices (determines size of deformation field)

        Returns:
            State with R=I, t=0, d=0
        """
        return cls(
            R=np.eye(3),
            t=np.zeros(3),
            d=np.zeros((n_vertices, 3)),
            cost=np.inf
        )

    @property
    def n_vertices(self) -> int:
        """Number of vertices this state applies to."""
        return len(self.d)

    def transform_vertices(self, vertices: np.ndarray) -> np.ndarray:
        """Apply transformation to vertices.

        Args:
            vertices: (N, 3) array

        Returns:
            (N, 3) transformed vertices: R @ v_i + d_i + t for each i

        Raises:
            ValueError: If number of vertices doesn't match deformation field size
        """
        vertices = np.asarray(vertices)
        if len(vertices) != len(self.d):
            raise ValueError(
                f"Number of vertices ({len(vertices)}) must match "
                f"deformation field size ({len(self.d)})"
            )

        # Apply rotation: (N, 3) @ (3, 3).T = (N, 3)
        rotated = vertices @ self.R.T
        # Add per-vertex deformation and global translation
        return rotated + self.d + self.t

    def copy(self) -> 'RegistrationState':
        """Create deep copy."""
        state = RegistrationState.__new__(RegistrationState)
        state.R = self.R.copy()
        state.t = self.t.copy()
        state.d = self.d.copy()
        state.cost = self.cost
        return state

    def __repr__(self) -> str:
        return (
            f"RegistrationState(n_vertices={self.n_vertices}, "
            f"cost={self.cost:.6f})"
        )
