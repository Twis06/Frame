import numpy as np
import matplotlib.pyplot as plt
import triangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

def convert_heightfield_to_trimesh(height_field_raw, horizontal_scale, vertical_scale, slope_threshold=None):
    """
    Convert a heightfield array to a triangle mesh represented by vertices and triangles.
    Optionally, corrects vertical surfaces above the provide slope threshold:

        If (y2-y1)/(x2-x1) > slope_threshold -> Move A to A' (set x1 = x2). Do this for all directions.
                   B(x2,y2)
                  /|
                 / |
                /  |
        (x1,y1)A---A'(x2',y1)

    Parameters:
        height_field_raw (np.array): input heightfield
        horizontal_scale (float): horizontal scale of the heightfield [meters]
        vertical_scale (float): vertical scale of the heightfield [meters]
        slope_threshold (float): the slope threshold above which surfaces are made vertical. If None no correction is applied (default: None)
    Returns:
        vertices (np.array(float)): array of shape (num_vertices, 3). Each row represents the location of each vertex [meters]
        triangles (np.array(int)): array of shape (num_triangles, 3). Each row represents the indices of the 3 vertices connected by this triangle.
    """
    hf = height_field_raw
    num_rows = hf.shape[0]
    num_cols = hf.shape[1]

    y = np.linspace(0, (num_cols-1)*horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows-1)*horizontal_scale, num_rows)
    yy, xx = np.meshgrid(y, x)

    if slope_threshold is not None:

        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows-1, :] += (hf[1:num_rows, :] - hf[:num_rows-1, :] > slope_threshold)
        move_x[1:num_rows, :] -= (hf[:num_rows-1, :] - hf[1:num_rows, :] > slope_threshold)
        move_y[:, :num_cols-1] += (hf[:, 1:num_cols] - hf[:, :num_cols-1] > slope_threshold)
        move_y[:, 1:num_cols] -= (hf[:, :num_cols-1] - hf[:, 1:num_cols] > slope_threshold)
        move_corners[:num_rows-1, :num_cols-1] += (hf[1:num_rows, 1:num_cols] - hf[:num_rows-1, :num_cols-1] > slope_threshold)
        move_corners[1:num_rows, 1:num_cols] -= (hf[:num_rows-1, :num_cols-1] - hf[1:num_rows, 1:num_cols] > slope_threshold)
        xx += (move_x + move_corners*(move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners*(move_y == 0)) * horizontal_scale

    # create triangle mesh vertices and triangles from the heightfield grid
    vertices = np.zeros((num_rows*num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2*(num_rows-1)*(num_cols-1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols-1) + i*num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2*i*(num_cols-1)
        stop = start + 2*(num_cols-1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start+1:stop:2, 0] = ind0
        triangles[start+1:stop:2, 1] = ind2
        triangles[start+1:stop:2, 2] = ind3

    return vertices, triangles, move_x != 0

def construct_window_outer_corners(inner_vertices, d=0.3):
    """
    Construct outer corners of a window based on inner vertices and frame width.
    Supports both XY plane and YZ plane.
    
    :param inner_vertices: np.array of shape (n, 3) representing inner corners of the window
    :param d: float, width of the window frame
    :return: np.array of shape (n, 3) representing outer corners of the window
    """
    # Determine the plane (XY or YZ)
    if np.allclose(inner_vertices[:, 2], 0):
        plane = 'xy'
        fixed_axis = 2
    elif np.allclose(inner_vertices[:, 0], 0):
        plane = 'yz'
        fixed_axis = 0
    else:
        raise ValueError("Vertices should be either in XY plane (z=0) or YZ plane (x=0)")

    # Get the two variable axes
    variable_axes = [i for i in range(3) if i != fixed_axis]

    # Extract 2D coordinates
    inner_2d = inner_vertices[:, variable_axes]

    # Calculate center of the inner vertices
    center = np.mean(inner_2d, axis=0)

    # Calculate vectors from center to each vertex
    vectors = inner_2d - center

    # Normalize vectors
    magnitudes = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / magnitudes

    # Calculate outer vertices in 2D
    outer_2d = inner_2d + normalized_vectors * d

    # Construct 3D outer vertices
    outer_vertices = np.zeros_like(inner_vertices)
    outer_vertices[:, variable_axes] = outer_2d
    outer_vertices[:, fixed_axis] = inner_vertices[:, fixed_axis]

    return outer_vertices

def convert_plane_window_to_trimesh(outer_vertices, inner_vertices):
    """
    Convert a 2D wall with a window (hole) to a triangle mesh using constrained Delaunay triangulation.
    Supports both XY plane (z=0) and YZ plane (x=0).
    """
    segments = np.vstack([
        np.column_stack([np.arange(len(outer_vertices)), np.roll(np.arange(len(outer_vertices)), -1)]),
        np.column_stack([np.arange(len(inner_vertices)) + len(outer_vertices),
                         np.roll(np.arange(len(inner_vertices)) + len(outer_vertices), -1)])
    ])

    # Determine which plane we're working on
    if np.allclose(outer_vertices[:, 2], 0) and np.allclose(inner_vertices[:, 2], 0):
        plane = 'xy'
        fixed_axis = 2
    elif np.allclose(outer_vertices[:, 0], 0) and np.allclose(inner_vertices[:, 0], 0):
        plane = 'yz'
        fixed_axis = 0
    else:
        raise ValueError('Vertices should be either in XY plane (z=0) or YZ plane (x=0)')

    # Select the two variable axes
    variable_axes = [i for i in range(3) if i != fixed_axis]
    vertices_2d = np.vstack([outer_vertices[:, variable_axes], inner_vertices[:, variable_axes]])

    tri_input = {
        'vertices': vertices_2d,
        'segments': segments,
        'holes': np.mean(inner_vertices[:, variable_axes], axis=0, keepdims=True)
    }

    tri_output = triangle.triangulate(tri_input, 'p')

    triangles = tri_output['triangles']
    new_vertices_2d = tri_output['vertices']

    new_vertices = np.zeros((len(new_vertices_2d), 3))
    new_vertices[:, variable_axes] = new_vertices_2d

    # Assign fixed-axis coordinates
    num_outer = len(outer_vertices)
    new_vertices[:num_outer, fixed_axis] = outer_vertices[:, fixed_axis]
    new_vertices[num_outer:len(vertices_2d), fixed_axis] = inner_vertices[:, fixed_axis]

    return new_vertices, triangles  # both should be n * 3

def visualize_mesh(outer_vertices, inner_vertices, vertices, triangles):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Add a small offset to z-coordinates to avoid singularity
    z_offset = 0.01
    vertices[:, 2] += np.linspace(0, z_offset, len(vertices))
    
    # Plot outer boundary
    ax.plot(outer_vertices[:, 0], outer_vertices[:, 1], outer_vertices[:, 2] + z_offset, 'r-', linewidth=2, label='Outer Boundary')
    ax.plot([outer_vertices[-1, 0], outer_vertices[0, 0]], 
            [outer_vertices[-1, 1], outer_vertices[0, 1]], 
            [outer_vertices[-1, 2] + z_offset, outer_vertices[0, 2] + z_offset], 'r-', linewidth=2)
    
    # Plot inner boundary (window)
    ax.plot(inner_vertices[:, 0], inner_vertices[:, 1], inner_vertices[:, 2] + z_offset, 'g-', linewidth=2, label='Window')
    ax.plot([inner_vertices[-1, 0], inner_vertices[0, 0]], 
            [inner_vertices[-1, 1], inner_vertices[0, 1]], 
            [inner_vertices[-1, 2] + z_offset, inner_vertices[0, 2] + z_offset], 'g-', linewidth=2)
    
    # Plot triangles
    mesh = Poly3DCollection([vertices[triangle] for triangle in triangles], alpha=0.3)
    mesh.set_edgecolor('b')
    ax.add_collection3d(mesh)
    
    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Wall with Window - Triangular Mesh')
    
    # Set axis limits
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(z_min - z_offset, z_max + z_offset)
    
    # Set equal aspect ratio
    ax.set_box_aspect((np.ptp(vertices[:, 0]), np.ptp(vertices[:, 1]), np.ptp(vertices[:, 2]) + 2*z_offset))
    
    # Adjust the viewing angle
    ax.view_init(elev=20, azim=45)
    
    plt.legend()
    plt.show()

def assemble_meshes(meshes, max_dist=10.0):
    """
      input: meshs = [(coordinates, triangles)] where the indices of each triangle is counted from 0
      This function combines meshes such that the coordinates have x-axis offsets and triangles have indices offsets
      """
    # extract vertices and triangles from meshes
    vertices_list = []
    triangles_list = []
    for i, mesh in enumerate(meshes):
        vertices = mesh[0].copy()  # create a copy to avoid modifying the original
        vertices[:, 0] += 2 * max_dist * i
        vertices_list.append(vertices)
        triangles_list.append(mesh[1])

    # concatenate all vertices
    vertices_array = np.concatenate(vertices_list, axis=0)

    # calculate the cumulative sum of vertices count
    vertices_count = np.array([mesh[0].shape[0] for mesh in meshes])
    vertices_offset = np.cumsum(vertices_count) - vertices_count

    # adjust triangle indices and concatenate
    adjusted_triangles = []
    for triangles, offset in zip(triangles_list, vertices_offset):
        adjusted_triangles.append(triangles + offset)
    triangles_array = np.concatenate(adjusted_triangles, axis=0)

    return vertices_array, triangles_array


if __name__ == '__main__':
    outer_vertices = np.array([
        [0, 0, 0], [0, 3, 0], [4, 3, 0], [5, 0, 0], [2, -1, 0]
    ])

    inner_vertices = np.array([
        [0.5, 1, 0], [1, 1.5, 0], [2, 1.5, 0], [3, 2, 0], [3.5, 1, 0], [4, 0, 0], [3, -0.5, 0], [2, 0, 0], [1, 1, 0], [0.5, 0, 0]
    ])

    vertices, triangles = convert_plane_window_to_trimesh(outer_vertices, inner_vertices)

    # Visualize the result
    visualize_mesh(outer_vertices, inner_vertices, vertices, triangles)