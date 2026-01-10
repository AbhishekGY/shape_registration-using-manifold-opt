"""Shape registration using manifold optimization on SO(3)."""

from .mesh import Mesh, RegistrationState
from .so3_manifold import (
    skew_part,
    symmetric_part,
    hat,
    vee,
    is_valid_rotation,
    rotation_error,
    project_so3_tangent,
    retract_so3,
    exp_so3,
    log_so3,
    geodesic_distance_so3,
    kabsch_algorithm,
    random_rotation,
    axis_angle_to_rotation,
    rotation_to_axis_angle,
)
from .cost_functions import (
    DataTerm,
    PointToPlaneDataTerm,
    SmoothnessRegularization,
    RigidityRegularization,
    compute_rotation_gradient,
    compute_translation_gradient,
    compute_deformation_gradient,
    numerical_gradient_check,
)
from .optimizer import (
    OptimizerConfig,
    OptimizationResult,
    ManifoldRegistration,
    create_optimizer,
)
from .visualization import (
    plot_mesh,
    plot_meshes,
    plot_registration_comparison,
    plot_cost_history,
    visualize_error_heatmap,
    visualize_registration_error,
    compute_vertex_errors,
    plot_correspondences,
    plot_gradient_norms,
    create_registration_summary,
)

__all__ = [
    # Data structures
    'Mesh',
    'RegistrationState',
    # SO(3) operations
    'skew_part',
    'symmetric_part',
    'hat',
    'vee',
    'is_valid_rotation',
    'rotation_error',
    'project_so3_tangent',
    'retract_so3',
    'exp_so3',
    'log_so3',
    'geodesic_distance_so3',
    'kabsch_algorithm',
    'random_rotation',
    'axis_angle_to_rotation',
    'rotation_to_axis_angle',
    # Cost functions
    'DataTerm',
    'PointToPlaneDataTerm',
    'SmoothnessRegularization',
    'RigidityRegularization',
    'compute_rotation_gradient',
    'compute_translation_gradient',
    'compute_deformation_gradient',
    'numerical_gradient_check',
    # Optimizer
    'OptimizerConfig',
    'OptimizationResult',
    'ManifoldRegistration',
    'create_optimizer',
    # Visualization
    'plot_mesh',
    'plot_meshes',
    'plot_registration_comparison',
    'plot_cost_history',
    'visualize_error_heatmap',
    'visualize_registration_error',
    'compute_vertex_errors',
    'plot_correspondences',
    'plot_gradient_norms',
    'create_registration_summary',
]
