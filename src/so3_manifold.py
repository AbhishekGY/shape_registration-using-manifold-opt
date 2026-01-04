"""SO(3) manifold operations for rotation optimization.

This module provides tools for optimizing over the special orthogonal group SO(3),
which is the group of 3x3 rotation matrices. SO(3) is a Lie group and a smooth
manifold, requiring special optimization techniques.

Key concepts:
- Tangent space at R: The set of matrices X such that R.T @ X is skew-symmetric
- Retraction: A map from tangent space back to the manifold (we use SVD projection)
- Exponential map: Another retraction using matrix exponential (exact but slower)
"""

import numpy as np
from typing import Tuple, Optional


def skew_part(M: np.ndarray) -> np.ndarray:
    """Extract skew-symmetric part of a matrix.

    For any square matrix M, we can decompose it as:
        M = skew(M) + sym(M)
    where skew(M) = (M - M.T) / 2

    A skew-symmetric matrix Omega satisfies Omega.T = -Omega,
    which means Omega[i,j] = -Omega[j,i] and diagonal is zero.

    Args:
        M: (n, n) square matrix

    Returns:
        (n, n) skew-symmetric matrix where result.T = -result
    """
    return 0.5 * (M - M.T)


def symmetric_part(M: np.ndarray) -> np.ndarray:
    """Extract symmetric part of a matrix.

    For any square matrix M:
        sym(M) = (M + M.T) / 2

    A symmetric matrix S satisfies S.T = S.

    Args:
        M: (n, n) square matrix

    Returns:
        (n, n) symmetric matrix where result.T = result
    """
    return 0.5 * (M + M.T)


def hat(omega: np.ndarray) -> np.ndarray:
    """Convert 3-vector to skew-symmetric matrix (hat operator).

    This is the Lie algebra isomorphism from R^3 to so(3).
    The resulting matrix [omega]_x satisfies: [omega]_x @ v = omega × v

    For omega = [w1, w2, w3]:
        [omega]_x = [[ 0,  -w3,  w2],
                     [ w3,  0,  -w1],
                     [-w2,  w1,  0 ]]

    Args:
        omega: (3,) vector

    Returns:
        (3, 3) skew-symmetric matrix
    """
    omega = np.asarray(omega).flatten()
    if len(omega) != 3:
        raise ValueError(f"omega must be a 3-vector, got shape {omega.shape}")

    return np.array([
        [0, -omega[2], omega[1]],
        [omega[2], 0, -omega[0]],
        [-omega[1], omega[0], 0]
    ], dtype=np.float64)


def vee(Omega: np.ndarray) -> np.ndarray:
    """Extract 3-vector from skew-symmetric matrix (vee operator).

    Inverse of the hat operator. Extracts the unique 3-vector omega
    such that hat(omega) = Omega.

    Args:
        Omega: (3, 3) skew-symmetric matrix

    Returns:
        (3,) vector
    """
    Omega = np.asarray(Omega)
    if Omega.shape != (3, 3):
        raise ValueError(f"Omega must be (3, 3), got {Omega.shape}")

    # Extract from off-diagonal elements
    # Omega[2,1] = w1, Omega[0,2] = w2, Omega[1,0] = w3
    return np.array([Omega[2, 1], Omega[0, 2], Omega[1, 0]], dtype=np.float64)


def is_valid_rotation(R: np.ndarray, tol: float = 1e-6) -> bool:
    """Check if matrix is a valid rotation matrix in SO(3).

    A matrix R is in SO(3) if:
        1. R is orthogonal: R.T @ R = I (columns are orthonormal)
        2. R has determinant +1 (not -1, which would be a reflection)

    Args:
        R: (3, 3) matrix to check
        tol: Tolerance for numerical errors

    Returns:
        True if R is in SO(3), False otherwise
    """
    if R.shape != (3, 3):
        return False

    # Check orthogonality: R.T @ R should be identity
    orthogonality_error = np.linalg.norm(R.T @ R - np.eye(3), ord='fro')
    if orthogonality_error > tol:
        return False

    # Check determinant: should be +1 (not -1)
    det = np.linalg.det(R)
    if not np.isclose(det, 1.0, atol=tol):
        return False

    return True


def rotation_error(R: np.ndarray) -> Tuple[float, float]:
    """Compute orthogonality and determinant errors for a matrix.

    Useful for debugging when rotations become invalid.

    Args:
        R: (3, 3) matrix

    Returns:
        orthogonality_error: ||R.T @ R - I||_F
        determinant_error: |det(R) - 1|
    """
    orthogonality_error = np.linalg.norm(R.T @ R - np.eye(3), ord='fro')
    determinant_error = abs(np.linalg.det(R) - 1.0)
    return orthogonality_error, determinant_error


def project_so3_tangent(R: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Project Euclidean gradient onto tangent space of SO(3) at R.

    The tangent space of SO(3) at R consists of matrices of the form R @ Omega,
    where Omega is skew-symmetric. Given a Euclidean gradient G (e.g., from
    differentiating a cost function), we project it onto this tangent space.

    The projection formula is derived from:
        G_tangent = R @ skew(R.T @ G)

    Which simplifies to:
        G_tangent = (G - R @ G.T @ R) / 2

    This ensures that R.T @ G_tangent is skew-symmetric, meaning G_tangent
    points along the manifold.

    Args:
        R: (3, 3) rotation matrix, must be in SO(3)
        G: (3, 3) Euclidean gradient matrix

    Returns:
        G_tangent: (3, 3) projected gradient in tangent space

    Note:
        The Riemannian gradient can be computed as:
            grad_R = G_tangent (for the canonical metric on SO(3))
    """
    # Method 1: Direct formula
    # G_tangent = 0.5 * (G - R @ G.T @ R)

    # Method 2: Via skew-symmetric component (equivalent, but clearer)
    # Omega = skew(R.T @ G)  # This is in the Lie algebra so(3)
    # G_tangent = R @ Omega

    # We use Method 1 as it's more efficient (fewer matrix multiplications)
    G_tangent = 0.5 * (G - R @ G.T @ R)

    return G_tangent


def retract_so3(M: np.ndarray) -> np.ndarray:
    """Project matrix onto SO(3) using SVD (closest rotation in Frobenius norm).

    Given an arbitrary 3x3 matrix M (typically R - lr * G_tangent from a gradient
    step), find the closest rotation matrix in Frobenius norm.

    The solution is given by the orthogonal Procrustes problem:
        R* = argmin_{R in SO(3)} ||R - M||_F

    Algorithm:
        1. Compute SVD: M = U @ S @ V.T
        2. Form R = U @ V.T (this gives closest orthogonal matrix)
        3. If det(R) = -1 (reflection), flip sign of last column of U
        4. Recompute R = U @ V.T

    Args:
        M: (3, 3) arbitrary matrix

    Returns:
        R: (3, 3) rotation matrix in SO(3)

    Note:
        This is the standard retraction for SO(3). For small steps, the
        exponential map (exp_so3) is more accurate but slower.
    """
    U, S, Vt = np.linalg.svd(M)
    R = U @ Vt

    # Ensure we get a rotation (det = +1), not a reflection (det = -1)
    if np.linalg.det(R) < 0:
        # Flip the sign of the last column of U
        U[:, -1] *= -1
        R = U @ Vt

    return R


def exp_so3(Omega: np.ndarray) -> np.ndarray:
    """Compute matrix exponential for skew-symmetric matrix (Rodrigues' formula).

    For a skew-symmetric matrix Omega = hat(omega), the exponential map
    gives a rotation matrix. This is computed efficiently using Rodrigues'
    rotation formula instead of general matrix exponential.

    For omega with ||omega|| = theta:
        exp(Omega) = I + (sin(theta)/theta) * Omega + ((1-cos(theta))/theta^2) * Omega^2

    Args:
        Omega: (3, 3) skew-symmetric matrix (element of so(3))

    Returns:
        R: (3, 3) rotation matrix in SO(3)
    """
    # Extract axis-angle representation
    omega = vee(Omega)
    theta = np.linalg.norm(omega)

    if theta < 1e-10:
        # Small angle approximation: exp(Omega) H I + Omega
        return np.eye(3) + Omega

    # Rodrigues' formula
    K = Omega / theta  # Normalized skew-symmetric matrix
    R = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)

    return R


def log_so3(R: np.ndarray) -> np.ndarray:
    """Compute matrix logarithm of rotation matrix (inverse of exp_so3).

    Given a rotation matrix R, find the skew-symmetric matrix Omega such that
    exp(Omega) = R. The rotation angle theta satisfies:
        trace(R) = 1 + 2*cos(theta)

    Args:
        R: (3, 3) rotation matrix in SO(3)

    Returns:
        Omega: (3, 3) skew-symmetric matrix in so(3)
    """
    # Compute rotation angle from trace
    cos_theta = (np.trace(R) - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)  # Numerical safety

    theta = np.arccos(cos_theta)

    if theta < 1e-10:
        # Small angle: log(R) H R - I (first order)
        return skew_part(R - np.eye(3))

    if abs(theta - np.pi) < 1e-6:
        # theta H pi: special case, extract axis from R + I
        # The axis is the eigenvector with eigenvalue 1
        eigvals, eigvecs = np.linalg.eig(R)
        idx = np.argmin(np.abs(eigvals - 1))
        axis = np.real(eigvecs[:, idx])
        axis = axis / np.linalg.norm(axis)
        return theta * hat(axis)

    # General case: use skew-symmetric part
    Omega = (theta / (2 * np.sin(theta))) * (R - R.T)

    return Omega


def geodesic_distance_so3(R1: np.ndarray, R2: np.ndarray) -> float:
    """Compute geodesic distance between two rotations on SO(3).

    The geodesic distance is the angle of the rotation R1.T @ R2,
    which equals ||log(R1.T @ R2)||_F / sqrt(2).

    Equivalently: d(R1, R2) = arccos((trace(R1.T @ R2) - 1) / 2)

    Args:
        R1: (3, 3) first rotation matrix
        R2: (3, 3) second rotation matrix

    Returns:
        Geodesic distance (rotation angle in radians)
    """
    R_diff = R1.T @ R2
    cos_theta = (np.trace(R_diff) - 1) / 2
    cos_theta = np.clip(cos_theta, -1, 1)
    return np.arccos(cos_theta)


def kabsch_algorithm(
    source: np.ndarray,
    target: np.ndarray,
    weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute optimal rotation and translation using Kabsch algorithm.

    Finds R, t that minimize the weighted sum of squared distances:
        sum_i w_i ||R @ source_i + t - target_i||^2

    This is a closed-form solution to the rigid alignment problem, useful for:
    - Initialization of more complex registration
    - Ground truth comparison for gradient-based methods
    - ICP inner loop

    Algorithm:
        1. Compute weighted centroids of both point sets
        2. Center both point sets
        3. Compute cross-covariance matrix H = sum_i w_i * source_i^T @ target_i
        4. SVD: H = U @ S @ V.T
        5. R = V @ U.T (with reflection correction)
        6. t = target_centroid - R @ source_centroid

    Args:
        source: (N, 3) source points to be transformed
        target: (N, 3) target points to align to
        weights: (N,) optional non-negative weights for each point pair.
                 If None, uniform weights are used.

    Returns:
        R: (3, 3) optimal rotation matrix
        t: (3,) optimal translation vector

    Raises:
        ValueError: If inputs have incompatible shapes
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    if source.shape != target.shape:
        raise ValueError(
            f"source and target must have same shape, "
            f"got {source.shape} and {target.shape}"
        )
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"source must have shape (N, 3), got {source.shape}")

    N = len(source)

    # Handle weights
    if weights is None:
        weights = np.ones(N)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (N,):
            raise ValueError(f"weights must have shape ({N},), got {weights.shape}")
        if np.any(weights < 0):
            raise ValueError("weights must be non-negative")

    # Normalize weights to sum to 1
    weight_sum = np.sum(weights)
    if weight_sum < 1e-10:
        raise ValueError("weights sum to zero")
    weights_normalized = weights / weight_sum

    # Compute weighted centroids
    source_centroid = np.sum(source * weights_normalized[:, np.newaxis], axis=0)
    target_centroid = np.sum(target * weights_normalized[:, np.newaxis], axis=0)

    # Center point clouds
    source_centered = source - source_centroid
    target_centered = target - target_centroid

    # Compute weighted cross-covariance matrix
    # H = sum_i w_i * source_centered[i].T @ target_centered[i]
    # Vectorized: H = source_centered.T @ diag(weights_normalized) @ target_centered
    H = source_centered.T @ (weights_normalized[:, np.newaxis] * target_centered)

    # SVD of cross-covariance
    U, S, Vt = np.linalg.svd(H)

    # Compute rotation: R = V @ U.T
    R = Vt.T @ U.T

    # Handle reflection case (det(R) = -1)
    if np.linalg.det(R) < 0:
        # Flip sign of last column of V (equivalently, last row of Vt)
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute translation
    t = target_centroid - R @ source_centroid

    return R, t


def random_rotation(rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Generate a uniformly random rotation matrix.

    Uses the QR decomposition method to generate a uniform distribution
    over SO(3) (Haar measure).

    Args:
        rng: NumPy random generator. If None, uses default.

    Returns:
        R: (3, 3) random rotation matrix
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate random 3x3 matrix with standard normal entries
    M = rng.standard_normal((3, 3))

    # QR decomposition
    Q, R_qr = np.linalg.qr(M)

    # Ensure positive diagonal in R_qr for uniqueness
    # Multiply Q columns by sign of R_qr diagonal
    signs = np.sign(np.diag(R_qr))
    signs[signs == 0] = 1
    Q = Q * signs

    # Ensure det = +1
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1

    return Q


def axis_angle_to_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Convert axis-angle representation to rotation matrix.

    Args:
        axis: (3,) unit vector defining rotation axis
        angle: Rotation angle in radians

    Returns:
        R: (3, 3) rotation matrix
    """
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)  # Ensure unit length

    Omega = angle * hat(axis)
    return exp_so3(Omega)


def rotation_to_axis_angle(R: np.ndarray) -> Tuple[np.ndarray, float]:
    """Convert rotation matrix to axis-angle representation.

    Args:
        R: (3, 3) rotation matrix

    Returns:
        axis: (3,) unit rotation axis
        angle: Rotation angle in radians [0, pi]
    """
    Omega = log_so3(R)
    omega = vee(Omega)
    angle = np.linalg.norm(omega)

    if angle < 1e-10:
        # No rotation - axis is arbitrary
        return np.array([1.0, 0.0, 0.0]), 0.0

    axis = omega / angle
    return axis, angle
