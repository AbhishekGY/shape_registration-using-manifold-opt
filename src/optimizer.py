"""Main registration optimizer using manifold optimization on SO(3).

This module implements the core optimization loop for 3D mesh registration,
combining rigid transformation (rotation + translation) with optional
non-rigid per-vertex deformations.

The optimization is performed on the product manifold SO(3) x R^3 x R^{3N},
with special handling for the rotation component using Riemannian gradient
descent on SO(3).
"""

import numpy as np
from typing import Optional, List, Tuple, Dict, Any, Callable
from dataclasses import dataclass, field

from .mesh import Mesh, RegistrationState
from .so3_manifold import (
    kabsch_algorithm,
    project_so3_tangent,
    retract_so3,
    is_valid_rotation,
    geodesic_distance_so3,
)
from .cost_functions import (
    DataTerm,
    PointToPlaneDataTerm,
    SmoothnessRegularization,
    compute_rotation_gradient,
    compute_translation_gradient,
    compute_deformation_gradient,
)


@dataclass
class OptimizerConfig:
    """Configuration for the registration optimizer."""

    # Learning rates
    lr_R: float = 0.01          # Rotation learning rate
    lr_t: float = 0.01          # Translation learning rate
    lr_d: float = 0.01          # Deformation learning rate (increase for more flexibility)

    # Regularization (lower = more deformation flexibility, higher = smoother)
    lambda_smooth: float = 0.01  # Smoothness regularization weight

    # Convergence criteria
    max_iterations: int = 300
    cost_tol: float = 1e-7      # Stop if |cost_change| < tol
    grad_tol: float = 1e-6      # Stop if gradient norm < tol

    # Correspondence options
    distance_threshold: Optional[float] = None  # Max correspondence distance
    update_correspondences_every: int = 1       # ICP-style updates

    # Optimization strategy
    optimize_rotation: bool = True
    optimize_translation: bool = True
    optimize_deformation: bool = True

    # Learning rate scheduling
    use_line_search: bool = False
    line_search_alpha: float = 0.5   # Backtracking factor
    line_search_max_iter: int = 10

    # Gradient clipping
    clip_grad_R: float = 1.0        # Max gradient norm for rotation
    clip_grad_t: float = 1.0        # Max gradient norm for translation
    clip_grad_d: float = 1.0        # Max gradient norm for deformation

    # Initialization
    init_method: str = 'kabsch'  # 'identity' or 'kabsch'

    # Verbosity
    verbose: bool = True
    print_every: int = 10


@dataclass
class OptimizationResult:
    """Result of registration optimization."""

    final_state: RegistrationState
    converged: bool
    n_iterations: int
    final_cost: float
    cost_history: List[float] = field(default_factory=list)
    rotation_history: List[np.ndarray] = field(default_factory=list)
    translation_history: List[np.ndarray] = field(default_factory=list)
    gradient_norms: List[Dict[str, float]] = field(default_factory=list)


class ManifoldRegistration:
    """Main registration optimizer using manifold optimization on SO(3).

    Minimizes the objective:
        L(R, t, d) = L_data(R, t, d) + lambda_smooth * L_smooth(d)

    where:
        - L_data: Point-to-point (or point-to-plane) distance
        - L_smooth: Deformation smoothness regularization
        - R in SO(3): Global rotation
        - t in R^3: Global translation
        - d in R^{3N}: Per-vertex deformation
    """

    def __init__(
        self,
        source_mesh: Mesh,
        target_mesh: Mesh,
        config: Optional[OptimizerConfig] = None,
        use_point_to_plane: bool = False
    ):
        """Initialize the registration optimizer.

        Args:
            source_mesh: Source mesh to be transformed
            target_mesh: Target mesh to align to
            config: Optimizer configuration (uses defaults if None)
            use_point_to_plane: Use point-to-plane distance (requires target normals)
        """
        self.source = source_mesh
        self.target = target_mesh
        self.config = config or OptimizerConfig()

        # Initialize cost functions
        if use_point_to_plane:
            target_normals = target_mesh.compute_vertex_normals()
            self.data_term = PointToPlaneDataTerm(
                target_mesh.vertices,
                target_normals
            )
        else:
            self.data_term = DataTerm(target_mesh.vertices)

        self.smoothness_reg = SmoothnessRegularization(
            source_mesh.get_edges(),
            weight=self.config.lambda_smooth
        )

        # Tracking
        self._cost_history: List[float] = []
        self._state_history: List[RegistrationState] = []
        self._gradient_norms: List[Dict[str, float]] = []

    def initialize_state(self) -> RegistrationState:
        """Initialize the registration state.

        Returns:
            Initial RegistrationState
        """
        N = self.source.n_vertices

        if self.config.init_method == 'identity':
            R = np.eye(3)
            t = np.zeros(3)
        elif self.config.init_method == 'kabsch':
            # Use Kabsch for initial rigid alignment
            R, t = kabsch_algorithm(
                self.source.vertices,
                self.target.vertices
            )
        elif self.config.init_method == 'centroid':
            # Just align centroids
            R = np.eye(3)
            t = self.target.center_of_mass() - self.source.center_of_mass()
        else:
            raise ValueError(f"Unknown init_method: {self.config.init_method}")

        d = np.zeros((N, 3))
        return RegistrationState(R, t, d)

    def compute_total_cost(self, state: RegistrationState) -> float:
        """Compute total objective value.

        Args:
            state: Current registration state

        Returns:
            Total cost (data + smoothness)
        """
        transformed = state.transform_vertices(self.source.vertices)

        # Data term
        data_cost = self.data_term.compute_cost(
            transformed,
            # weights could be added here
        )

        # Smoothness term
        smooth_cost = self.smoothness_reg.compute_cost(state.d)

        return data_cost + smooth_cost

    def compute_gradients(
        self,
        state: RegistrationState
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute gradients w.r.t. R, t, and d.

        Args:
            state: Current registration state

        Returns:
            G_R: (3, 3) Euclidean gradient w.r.t. R
            g_t: (3,) gradient w.r.t. translation
            g_d: (N, 3) gradient w.r.t. deformation
        """
        transformed = state.transform_vertices(self.source.vertices)

        # Gradient of data term w.r.t. transformed vertices
        grad_data = self.data_term.compute_gradient_wrt_transformed(transformed)

        # Gradient w.r.t. rotation (Euclidean, not yet projected)
        G_R = compute_rotation_gradient(
            self.source.vertices,
            state.R,
            grad_data
        )

        # Gradient w.r.t. translation
        g_t = compute_translation_gradient(grad_data)

        # Gradient w.r.t. deformation (data + smoothness)
        g_d = compute_deformation_gradient(
            grad_data,
            self.smoothness_reg,
            state.d
        )

        return G_R, g_t, g_d

    def _clip_gradient(self, grad: np.ndarray, max_norm: float) -> np.ndarray:
        """Clip gradient to maximum norm.

        Args:
            grad: Gradient array
            max_norm: Maximum allowed norm

        Returns:
            Clipped gradient
        """
        norm = np.linalg.norm(grad)
        if norm > max_norm and norm > 0:
            return grad * (max_norm / norm)
        return grad

    def _gradient_step(
        self,
        state: RegistrationState,
        G_R: np.ndarray,
        g_t: np.ndarray,
        g_d: np.ndarray,
        lr_R: float,
        lr_t: float,
        lr_d: float
    ) -> RegistrationState:
        """Take a gradient descent step.

        Args:
            state: Current state
            G_R, g_t, g_d: Gradients
            lr_R, lr_t, lr_d: Learning rates

        Returns:
            New state after gradient step
        """
        # Create new state (copy to avoid modifying original)
        new_R = state.R.copy()
        new_t = state.t.copy()
        new_d = state.d.copy()

        # Clip gradients
        G_R_clipped = self._clip_gradient(G_R, self.config.clip_grad_R)
        g_t_clipped = self._clip_gradient(g_t, self.config.clip_grad_t)
        g_d_clipped = self._clip_gradient(g_d, self.config.clip_grad_d)

        # Update rotation (manifold gradient descent)
        if self.config.optimize_rotation:
            # Project gradient to tangent space of SO(3)
            G_R_tangent = project_so3_tangent(state.R, G_R_clipped)
            # Gradient step in ambient space
            R_updated = state.R - lr_R * G_R_tangent
            # Retract back to SO(3)
            new_R = retract_so3(R_updated)

        # Update translation (Euclidean gradient descent)
        if self.config.optimize_translation:
            new_t = state.t - lr_t * g_t_clipped

        # Update deformation (Euclidean gradient descent)
        if self.config.optimize_deformation:
            new_d = state.d - lr_d * g_d_clipped

        # Validate rotation
        if not is_valid_rotation(new_R):
            raise RuntimeError("Rotation became invalid after update!")

        return RegistrationState(new_R, new_t, new_d)

    def _line_search(
        self,
        state: RegistrationState,
        G_R: np.ndarray,
        g_t: np.ndarray,
        g_d: np.ndarray,
        current_cost: float
    ) -> Tuple[RegistrationState, float, float]:
        """Backtracking line search to find good step size.

        Args:
            state: Current state
            G_R, g_t, g_d: Gradients
            current_cost: Current objective value

        Returns:
            new_state: State after line search
            new_cost: Cost at new state
            step_size: Final step size multiplier
        """
        alpha = self.config.line_search_alpha
        step_size = 1.0

        for _ in range(self.config.line_search_max_iter):
            # Try step with current step size
            new_state = self._gradient_step(
                state, G_R, g_t, g_d,
                lr_R=self.config.lr_R * step_size,
                lr_t=self.config.lr_t * step_size,
                lr_d=self.config.lr_d * step_size
            )
            new_cost = self.compute_total_cost(new_state)

            # Check Armijo condition (sufficient decrease)
            if new_cost < current_cost:
                return new_state, new_cost, step_size

            step_size *= alpha

        # Return last attempt even if no decrease
        return new_state, new_cost, step_size

    def optimize(self) -> OptimizationResult:
        """Run the optimization loop.

        Returns:
            OptimizationResult containing final state and history
        """
        # Initialize
        state = self.initialize_state()
        cost = self.compute_total_cost(state)

        self._cost_history = [cost]
        self._state_history = [state.copy()]
        self._gradient_norms = []

        converged = False
        iteration = 0

        if self.config.verbose:
            print(f"Initial cost: {cost:.6f}")

        for iteration in range(self.config.max_iterations):
            # Compute gradients
            G_R, g_t, g_d = self.compute_gradients(state)

            # Track gradient norms
            grad_norms = {
                'R': np.linalg.norm(G_R),
                't': np.linalg.norm(g_t),
                'd': np.linalg.norm(g_d),
            }
            self._gradient_norms.append(grad_norms)

            # Check gradient convergence
            total_grad_norm = np.sqrt(
                grad_norms['R']**2 + grad_norms['t']**2 + grad_norms['d']**2
            )
            if total_grad_norm < self.config.grad_tol:
                if self.config.verbose:
                    print(f"Converged: gradient norm {total_grad_norm:.2e} < {self.config.grad_tol}")
                converged = True
                break

            # Take gradient step
            if self.config.use_line_search:
                new_state, new_cost, step_size = self._line_search(
                    state, G_R, g_t, g_d, cost
                )
            else:
                new_state = self._gradient_step(
                    state, G_R, g_t, g_d,
                    self.config.lr_R,
                    self.config.lr_t,
                    self.config.lr_d
                )
                new_cost = self.compute_total_cost(new_state)

            # Check cost convergence
            cost_change = abs(new_cost - cost)
            if cost_change < self.config.cost_tol:
                if self.config.verbose:
                    print(f"Converged: cost change {cost_change:.2e} < {self.config.cost_tol}")
                converged = True
                state = new_state
                cost = new_cost
                self._cost_history.append(cost)
                self._state_history.append(state.copy())
                break

            # Update state
            state = new_state
            cost = new_cost
            self._cost_history.append(cost)
            self._state_history.append(state.copy())

            # Print progress
            if self.config.verbose and (iteration + 1) % self.config.print_every == 0:
                print(f"Iter {iteration + 1:4d}: cost = {cost:.6f}, "
                      f"|grad| = {total_grad_norm:.2e}")

        if self.config.verbose and not converged:
            print(f"Max iterations ({self.config.max_iterations}) reached")

        # Final state
        state.cost = cost

        return OptimizationResult(
            final_state=state,
            converged=converged,
            n_iterations=iteration + 1,
            final_cost=cost,
            cost_history=self._cost_history.copy(),
            rotation_history=[s.R for s in self._state_history],
            translation_history=[s.t for s in self._state_history],
            gradient_norms=self._gradient_norms.copy(),
        )

    def optimize_icp_style(
        self,
        n_outer_iterations: int = 10
    ) -> OptimizationResult:
        """Run ICP-style optimization with correspondence updates.

        This alternates between:
        1. Finding correspondences (nearest neighbors)
        2. Optimizing transformation given fixed correspondences

        Args:
            n_outer_iterations: Number of correspondence update cycles

        Returns:
            OptimizationResult
        """
        # Save original config
        original_max_iter = self.config.max_iterations
        original_verbose = self.config.verbose

        # Use fewer iterations per inner loop
        inner_iterations = max(10, original_max_iter // n_outer_iterations)
        self.config.max_iterations = inner_iterations
        self.config.verbose = False

        # Initialize
        state = self.initialize_state()
        all_cost_history = []
        all_gradient_norms = []

        if original_verbose:
            print("Running ICP-style optimization...")

        for outer_iter in range(n_outer_iterations):
            # Run inner optimization
            result = self.optimize()

            # Accumulate history
            all_cost_history.extend(result.cost_history)
            all_gradient_norms.extend(result.gradient_norms)

            if original_verbose:
                print(f"Outer iter {outer_iter + 1}/{n_outer_iterations}: "
                      f"cost = {result.final_cost:.6f}")

            # Update state for next iteration
            state = result.final_state

            # Check if converged
            if result.converged and len(result.cost_history) < inner_iterations // 2:
                if original_verbose:
                    print("Early convergence in ICP loop")
                break

            # Re-initialize optimizer state for next round
            # (correspondences will be recomputed automatically)

        # Restore config
        self.config.max_iterations = original_max_iter
        self.config.verbose = original_verbose

        return OptimizationResult(
            final_state=state,
            converged=result.converged,
            n_iterations=len(all_cost_history),
            final_cost=result.final_cost,
            cost_history=all_cost_history,
            rotation_history=result.rotation_history,
            translation_history=result.translation_history,
            gradient_norms=all_gradient_norms,
        )

    @property
    def cost_history(self) -> List[float]:
        """Get cost history from last optimization."""
        return self._cost_history

    @property
    def state_history(self) -> List[RegistrationState]:
        """Get state history from last optimization."""
        return self._state_history

    def get_aligned_mesh(self, state: Optional[RegistrationState] = None) -> Mesh:
        """Get the source mesh transformed by the given state.

        Args:
            state: Registration state to apply. If None, uses last state from history.

        Returns:
            Transformed source mesh
        """
        if state is None:
            if not self._state_history:
                raise ValueError("No optimization has been run yet")
            state = self._state_history[-1]

        transformed_vertices = state.transform_vertices(self.source.vertices)
        return Mesh(transformed_vertices, self.source.faces.copy())

    def compute_alignment_error(
        self,
        state: Optional[RegistrationState] = None
    ) -> Dict[str, float]:
        """Compute alignment error statistics.

        Args:
            state: State to evaluate. If None, uses last state.

        Returns:
            Dictionary with error statistics
        """
        if state is None:
            if not self._state_history:
                raise ValueError("No optimization has been run yet")
            state = self._state_history[-1]

        transformed = state.transform_vertices(self.source.vertices)

        # Get distances to nearest target points
        _, _, mask = self.data_term.compute_correspondences(transformed)
        distances = self.data_term.get_correspondence_distances()

        if distances is None:
            distances = np.zeros(len(transformed))

        valid_distances = distances[mask] if mask.any() else distances

        return {
            'mean': float(np.mean(valid_distances)),
            'median': float(np.median(valid_distances)),
            'max': float(np.max(valid_distances)),
            'std': float(np.std(valid_distances)),
            'rmse': float(np.sqrt(np.mean(valid_distances**2))),
            'n_valid': int(np.sum(mask)),
            'n_total': len(distances),
        }


def create_optimizer(
    source: Mesh,
    target: Mesh,
    lambda_smooth: float = 0.1,
    lr_R: float = 0.01,
    lr_t: float = 0.01,
    lr_d: float = 0.001,
    max_iterations: int = 200,
    use_point_to_plane: bool = False,
    verbose: bool = True
) -> ManifoldRegistration:
    """Convenience function to create a configured optimizer.

    Args:
        source: Source mesh
        target: Target mesh
        lambda_smooth: Smoothness regularization weight
        lr_R, lr_t, lr_d: Learning rates
        max_iterations: Maximum optimization iterations
        use_point_to_plane: Use point-to-plane distance
        verbose: Print progress

    Returns:
        Configured ManifoldRegistration optimizer
    """
    config = OptimizerConfig(
        lr_R=lr_R,
        lr_t=lr_t,
        lr_d=lr_d,
        lambda_smooth=lambda_smooth,
        max_iterations=max_iterations,
        verbose=verbose,
    )

    return ManifoldRegistration(
        source,
        target,
        config=config,
        use_point_to_plane=use_point_to_plane
    )
