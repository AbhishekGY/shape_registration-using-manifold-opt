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
]
