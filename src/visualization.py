"""Visualization functions for mesh registration.

This module provides plotting utilities for:
- 3D mesh visualization
- Registration comparison (source, target, aligned)
- Optimization convergence plots
- Error heatmaps on mesh surfaces
- Correspondence visualization
"""

import numpy as np
from typing import Optional, List, Tuple, Union
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.colors as mcolors

from .mesh import Mesh, RegistrationState


def _set_axes_equal(ax: plt.Axes):
    """Set equal aspect ratio for 3D axes.

    Matplotlib 3D doesn't have equal aspect by default, so we compute
    the bounding box and set limits manually.

    Args:
        ax: Matplotlib 3D axis
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])

    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))

    ax.set_xlim3d([origin[0] - radius, origin[0] + radius])
    ax.set_ylim3d([origin[1] - radius, origin[1] + radius])
    ax.set_zlim3d([origin[2] - radius, origin[2] + radius])


def plot_mesh(
    mesh: Mesh,
    ax: Optional[plt.Axes] = None,
    color: str = 'cyan',
    alpha: float = 0.6,
    edge_color: str = 'black',
    edge_alpha: float = 0.2,
    show_edges: bool = True,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Axes:
    """Plot a 3D mesh.

    Args:
        mesh: Mesh to plot
        ax: Matplotlib 3D axis (creates new if None)
        color: Face color
        alpha: Face transparency (0-1)
        edge_color: Edge color
        edge_alpha: Edge transparency
        show_edges: Whether to draw edges
        title: Plot title
        figsize: Figure size if creating new figure

    Returns:
        Matplotlib 3D axis
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    # Create triangles from vertices and faces
    triangles = mesh.vertices[mesh.faces]

    # Create Poly3DCollection
    collection = Poly3DCollection(
        triangles,
        alpha=alpha,
        facecolor=color,
        edgecolor=edge_color if show_edges else 'none',
        linewidths=0.5 if show_edges else 0,
    )
    if show_edges:
        collection.set_edgecolor((0, 0, 0, edge_alpha))

    ax.add_collection3d(collection)

    # Set axis limits based on mesh bounds
    vertices = mesh.vertices
    margin = 0.1 * mesh.bounding_box_diagonal()
    ax.set_xlim(vertices[:, 0].min() - margin, vertices[:, 0].max() + margin)
    ax.set_ylim(vertices[:, 1].min() - margin, vertices[:, 1].max() + margin)
    ax.set_zlim(vertices[:, 2].min() - margin, vertices[:, 2].max() + margin)

    _set_axes_equal(ax)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if title:
        ax.set_title(title)

    return ax


def plot_meshes(
    meshes: List[Mesh],
    colors: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    alphas: Optional[List[float]] = None,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 10),
    show_legend: bool = True
) -> plt.Axes:
    """Plot multiple meshes overlaid.

    Args:
        meshes: List of meshes to plot
        colors: List of colors for each mesh
        labels: List of labels for legend
        alphas: List of alpha values
        ax: Matplotlib 3D axis
        title: Plot title
        figsize: Figure size
        show_legend: Whether to show legend

    Returns:
        Matplotlib 3D axis
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    n_meshes = len(meshes)

    # Default colors
    if colors is None:
        default_colors = ['red', 'blue', 'green', 'orange', 'purple', 'cyan']
        colors = [default_colors[i % len(default_colors)] for i in range(n_meshes)]

    # Default alphas
    if alphas is None:
        alphas = [0.5] * n_meshes

    # Default labels
    if labels is None:
        labels = [f'Mesh {i+1}' for i in range(n_meshes)]

    # Plot each mesh
    legend_handles = []
    for mesh, color, alpha, label in zip(meshes, colors, alphas, labels):
        triangles = mesh.vertices[mesh.faces]
        collection = Poly3DCollection(
            triangles,
            alpha=alpha,
            facecolor=color,
            edgecolor=(0, 0, 0, 0.1),
            linewidths=0.3,
        )
        ax.add_collection3d(collection)

        # Create proxy artist for legend
        from matplotlib.patches import Patch
        legend_handles.append(Patch(facecolor=color, alpha=alpha, label=label))

    # Set limits based on all meshes
    all_vertices = np.vstack([m.vertices for m in meshes])
    margin = 0.1 * np.max(np.ptp(all_vertices, axis=0))
    ax.set_xlim(all_vertices[:, 0].min() - margin, all_vertices[:, 0].max() + margin)
    ax.set_ylim(all_vertices[:, 1].min() - margin, all_vertices[:, 1].max() + margin)
    ax.set_zlim(all_vertices[:, 2].min() - margin, all_vertices[:, 2].max() + margin)

    _set_axes_equal(ax)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    if show_legend:
        ax.legend(handles=legend_handles, loc='upper left')

    if title:
        ax.set_title(title)

    return ax


def plot_registration_comparison(
    source: Mesh,
    target: Mesh,
    aligned: Mesh,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 6),
    show: bool = True
) -> plt.Figure:
    """Plot source, target, and aligned meshes side-by-side.

    Creates a three-panel figure showing:
    1. Original source mesh
    2. Target mesh
    3. Aligned source overlaid on target

    Args:
        source: Original source mesh
        target: Target mesh
        aligned: Registered/aligned source mesh
        save_path: Optional path to save figure
        figsize: Figure size
        show: Whether to display the figure

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)

    # Source mesh
    ax1 = fig.add_subplot(131, projection='3d')
    plot_mesh(source, ax=ax1, color='red', alpha=0.7, title='Source Mesh')

    # Target mesh
    ax2 = fig.add_subplot(132, projection='3d')
    plot_mesh(target, ax=ax2, color='blue', alpha=0.7, title='Target Mesh')

    # Aligned overlay
    ax3 = fig.add_subplot(133, projection='3d')
    plot_meshes(
        [aligned, target],
        colors=['green', 'blue'],
        labels=['Aligned', 'Target'],
        alphas=[0.7, 0.3],
        ax=ax3,
        title='Aligned vs Target'
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def plot_cost_history(
    cost_history: List[float],
    log_scale: bool = True,
    title: str = 'Optimization Convergence',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    show: bool = True
) -> plt.Figure:
    """Plot cost vs iteration.

    Args:
        cost_history: List of cost values per iteration
        log_scale: Use log scale for y-axis
        title: Plot title
        save_path: Optional path to save figure
        figsize: Figure size
        show: Whether to display

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    iterations = np.arange(len(cost_history))
    ax.plot(iterations, cost_history, linewidth=2, color='blue', marker='o',
            markersize=3, markevery=max(1, len(cost_history) // 20))

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Cost', fontsize=12)
    ax.set_title(title, fontsize=14)

    if log_scale and min(cost_history) > 0:
        ax.set_yscale('log')

    ax.grid(True, alpha=0.3)

    # Add annotations
    ax.annotate(f'Initial: {cost_history[0]:.4f}',
                xy=(0, cost_history[0]),
                xytext=(10, 20), textcoords='offset points',
                fontsize=10, alpha=0.8)
    ax.annotate(f'Final: {cost_history[-1]:.4f}',
                xy=(len(cost_history)-1, cost_history[-1]),
                xytext=(-60, 20), textcoords='offset points',
                fontsize=10, alpha=0.8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def visualize_error_heatmap(
    mesh: Mesh,
    errors: np.ndarray,
    title: str = 'Alignment Error',
    cmap: str = 'hot',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
    show: bool = True,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None
) -> plt.Figure:
    """Visualize per-vertex error as heatmap on mesh surface.

    Args:
        mesh: Mesh to visualize
        errors: (N,) array of per-vertex error values
        title: Plot title
        cmap: Matplotlib colormap name
        save_path: Optional path to save figure
        figsize: Figure size
        show: Whether to display
        vmin: Minimum value for colormap
        vmax: Maximum value for colormap

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Compute face colors by averaging vertex errors
    face_errors = np.mean(errors[mesh.faces], axis=1)

    # Normalize colors
    if vmin is None:
        vmin = errors.min()
    if vmax is None:
        vmax = errors.max()

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colormap = plt.cm.get_cmap(cmap)
    face_colors = colormap(norm(face_errors))

    # Create triangles
    triangles = mesh.vertices[mesh.faces]

    # Create collection with per-face colors
    collection = Poly3DCollection(
        triangles,
        facecolors=face_colors,
        edgecolors=(0, 0, 0, 0.1),
        linewidths=0.2,
    )
    ax.add_collection3d(collection)

    # Set limits
    vertices = mesh.vertices
    margin = 0.1 * mesh.bounding_box_diagonal()
    ax.set_xlim(vertices[:, 0].min() - margin, vertices[:, 0].max() + margin)
    ax.set_ylim(vertices[:, 1].min() - margin, vertices[:, 1].max() + margin)
    ax.set_zlim(vertices[:, 2].min() - margin, vertices[:, 2].max() + margin)

    _set_axes_equal(ax)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)

    # Add colorbar
    mappable = plt.cm.ScalarMappable(norm=norm, cmap=colormap)
    mappable.set_array(errors)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.6, pad=0.1)
    cbar.set_label('Error', fontsize=10)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def compute_vertex_errors(
    transformed_source: np.ndarray,
    target_vertices: np.ndarray
) -> np.ndarray:
    """Compute per-vertex distance to nearest target point.

    Args:
        transformed_source: (N, 3) transformed source vertices
        target_vertices: (M, 3) target vertices

    Returns:
        (N,) distances to nearest target point
    """
    from scipy.spatial import KDTree
    kdtree = KDTree(target_vertices)
    distances, _ = kdtree.query(transformed_source)
    return distances


def visualize_registration_error(
    source: Mesh,
    target: Mesh,
    state: RegistrationState,
    title: str = 'Registration Error Heatmap',
    save_path: Optional[str] = None,
    show: bool = True
) -> Tuple[plt.Figure, dict]:
    """Visualize registration error on aligned source mesh.

    Args:
        source: Source mesh
        target: Target mesh
        state: Registration state
        title: Plot title
        save_path: Optional save path
        show: Whether to display

    Returns:
        Figure and error statistics dictionary
    """
    # Transform source vertices
    transformed = state.transform_vertices(source.vertices)

    # Compute errors
    errors = compute_vertex_errors(transformed, target.vertices)

    # Create aligned mesh for visualization
    aligned_mesh = Mesh(transformed, source.faces.copy())

    # Visualize
    fig = visualize_error_heatmap(
        aligned_mesh,
        errors,
        title=title,
        save_path=save_path,
        show=show
    )

    # Compute statistics
    stats = {
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'max': float(np.max(errors)),
        'min': float(np.min(errors)),
        'std': float(np.std(errors)),
        'rmse': float(np.sqrt(np.mean(errors**2))),
    }

    return fig, stats


def plot_correspondences(
    source_vertices: np.ndarray,
    target_vertices: np.ndarray,
    correspondences: np.ndarray,
    ax: Optional[plt.Axes] = None,
    sample_ratio: float = 0.1,
    source_color: str = 'red',
    target_color: str = 'blue',
    line_color: str = 'gray',
    figsize: Tuple[int, int] = (10, 10),
    title: str = 'Point Correspondences'
) -> plt.Axes:
    """Visualize point correspondences between source and target.

    Args:
        source_vertices: (N, 3) source points
        target_vertices: (M, 3) target points
        correspondences: (N,) indices into target for each source
        ax: Matplotlib 3D axis
        sample_ratio: Fraction of correspondences to show (for clarity)
        source_color: Color for source points
        target_color: Color for target points
        line_color: Color for correspondence lines
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib 3D axis
    """
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d')

    # Sample correspondences for clarity
    n_samples = max(1, int(len(source_vertices) * sample_ratio))
    indices = np.random.choice(len(source_vertices), n_samples, replace=False)

    # Plot source points
    ax.scatter(
        source_vertices[indices, 0],
        source_vertices[indices, 1],
        source_vertices[indices, 2],
        c=source_color, s=20, alpha=0.8, label='Source'
    )

    # Plot target points (corresponding)
    target_indices = correspondences[indices]
    ax.scatter(
        target_vertices[target_indices, 0],
        target_vertices[target_indices, 1],
        target_vertices[target_indices, 2],
        c=target_color, s=20, alpha=0.8, label='Target'
    )

    # Draw correspondence lines
    for i in indices:
        src = source_vertices[i]
        tgt = target_vertices[correspondences[i]]
        ax.plot(
            [src[0], tgt[0]],
            [src[1], tgt[1]],
            [src[2], tgt[2]],
            c=line_color, alpha=0.3, linewidth=0.5
        )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()

    _set_axes_equal(ax)

    return ax


def plot_gradient_norms(
    gradient_norms: List[dict],
    title: str = 'Gradient Norms During Optimization',
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 6),
    show: bool = True
) -> plt.Figure:
    """Plot gradient norms over iterations.

    Args:
        gradient_norms: List of dicts with 'R', 't', 'd' keys
        title: Plot title
        save_path: Optional save path
        figsize: Figure size
        show: Whether to display

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    iterations = np.arange(len(gradient_norms))

    # Extract gradient norms
    R_norms = [g.get('R', 0) for g in gradient_norms]
    t_norms = [g.get('t', 0) for g in gradient_norms]
    d_norms = [g.get('d', 0) for g in gradient_norms]

    ax.semilogy(iterations, R_norms, label='Rotation', linewidth=2)
    ax.semilogy(iterations, t_norms, label='Translation', linewidth=2)
    ax.semilogy(iterations, d_norms, label='Deformation', linewidth=2)

    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Gradient Norm (log)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig


def create_registration_summary(
    source: Mesh,
    target: Mesh,
    result,  # OptimizationResult
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Create a comprehensive summary figure for registration results.

    Shows:
    - Original meshes
    - Aligned result
    - Cost convergence
    - Error heatmap

    Args:
        source: Source mesh
        target: Target mesh
        result: OptimizationResult from optimizer
        save_path: Optional save path
        show: Whether to display

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(16, 12))

    # Transform source
    transformed = result.final_state.transform_vertices(source.vertices)
    aligned = Mesh(transformed, source.faces.copy())

    # 1. Source mesh
    ax1 = fig.add_subplot(231, projection='3d')
    plot_mesh(source, ax=ax1, color='red', alpha=0.7, title='Source')

    # 2. Target mesh
    ax2 = fig.add_subplot(232, projection='3d')
    plot_mesh(target, ax=ax2, color='blue', alpha=0.7, title='Target')

    # 3. Aligned overlay
    ax3 = fig.add_subplot(233, projection='3d')
    plot_meshes(
        [aligned, target],
        colors=['green', 'blue'],
        alphas=[0.7, 0.3],
        ax=ax3,
        title='Aligned vs Target',
        show_legend=True
    )

    # 4. Cost history
    ax4 = fig.add_subplot(234)
    iterations = np.arange(len(result.cost_history))
    ax4.plot(iterations, result.cost_history, linewidth=2)
    ax4.set_xlabel('Iteration')
    ax4.set_ylabel('Cost')
    ax4.set_title('Convergence')
    ax4.grid(True, alpha=0.3)
    if min(result.cost_history) > 0:
        ax4.set_yscale('log')

    # 5. Error heatmap
    ax5 = fig.add_subplot(235, projection='3d')
    errors = compute_vertex_errors(transformed, target.vertices)
    face_errors = np.mean(errors[aligned.faces], axis=1)
    norm = mcolors.Normalize(vmin=errors.min(), vmax=errors.max())
    colormap = plt.cm.get_cmap('hot')
    face_colors = colormap(norm(face_errors))
    triangles = aligned.vertices[aligned.faces]
    collection = Poly3DCollection(triangles, facecolors=face_colors, edgecolors=(0,0,0,0.1), linewidths=0.2)
    ax5.add_collection3d(collection)
    vertices = aligned.vertices
    margin = 0.1 * aligned.bounding_box_diagonal()
    ax5.set_xlim(vertices[:, 0].min() - margin, vertices[:, 0].max() + margin)
    ax5.set_ylim(vertices[:, 1].min() - margin, vertices[:, 1].max() + margin)
    ax5.set_zlim(vertices[:, 2].min() - margin, vertices[:, 2].max() + margin)
    _set_axes_equal(ax5)
    ax5.set_title('Error Heatmap')

    # 6. Statistics text
    ax6 = fig.add_subplot(236)
    ax6.axis('off')
    stats_text = f"""Registration Summary

Converged: {result.converged}
Iterations: {result.n_iterations}
Final Cost: {result.final_cost:.6f}

Error Statistics:
  Mean:   {np.mean(errors):.6f}
  Median: {np.median(errors):.6f}
  Max:    {np.max(errors):.6f}
  RMSE:   {np.sqrt(np.mean(errors**2)):.6f}

Cost Reduction:
  Initial: {result.cost_history[0]:.6f}
  Final:   {result.cost_history[-1]:.6f}
  Ratio:   {result.cost_history[0]/max(result.cost_history[-1], 1e-10):.2f}x
"""
    ax6.text(0.1, 0.9, stats_text, transform=ax6.transAxes,
             fontsize=11, verticalalignment='top', fontfamily='monospace')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    if show:
        plt.show()

    return fig
