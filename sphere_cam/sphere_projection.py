import torch
import numpy as np

def deproject_pointcloud(depth_image, intrinsic, extrinsic=None, device=None):
    """
    Converts a single 2D depth image into a 3D point cloud using camera intrinsic
    and optional extrinsic matrices.

    Args:
        depth_image (torch.Tensor): A 2D tensor of shape (height, width) where each value represents depth in meters.
        intrinsic (torch.Tensor): The camera's intrinsic matrix of shape (3, 3), which includes the focal lengths and 
                                  principal point offsets.
        extrinsic (torch.Tensor, optional): The camera's extrinsic matrix of shape (4, 4) that transforms points from the 
                                            camera coordinate system to the world coordinate system (default is None, meaning 
                                            points remain in the camera coordinate system).
        device (torch.device, optional): The device to use for computations (e.g., 'cuda' or 'cpu'). If None, defaults to the
                                         device of the depth_image.

    Returns:
        torch.Tensor: A tensor of shape (height, width, 3), where each pixel corresponds to a 3D point (x, y, z) in space.
    """
    
    # Set the computation device (use the depth image's device if none is provided)
    if device is None:
        device = depth_image.device
    
    # Get the height and width of the depth image
    height, width = depth_image.shape
    
    # Initialize a tensor to store the final 3D points, of shape (height, width, 3)
    points_3d = torch.zeros((height, width, 3), device=device)
    
    # Extract focal lengths (fx, fy) and principal point offsets (cx, cy) from the intrinsic matrix
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    
    # Create a grid of pixel coordinates for the depth image (height x width)
    u_coords, v_coords = torch.meshgrid(torch.arange(0, height, device=device), 
                                        torch.arange(0, width, device=device), 
                                        indexing='ij')
    
    # Normalize the pixel coordinates using the intrinsic parameters
    # Convert (u, v) image coordinates to (x, y) in normalized camera coordinates
    x_norm = (v_coords - cx) / fx  # Normalized x coordinates
    y_norm = (u_coords - cy) / fy  # Normalized y coordinates

    y_norm = -y_norm
    x_norm = -x_norm

    # Set the infinity values (20) to 0
    # depth_image = torch.where(depth_image==20,0,depth_image)
    
    # Compute the 3D x and y coordinates at unit depth
    # Scale x, y by the actual depth from the depth_image
    z_coords = depth_image  # z is the depth value for each pixel
    x_coords = x_norm * z_coords  # x = x_norm * depth
    y_coords = y_norm * z_coords  # y = y_norm * depth

    # Combine the x, y, z coordinates into the points_3d tensor
    # Change to world coordinate system, x forward, z up, y left
    points_3d[..., 0] = z_coords  # Assign z camera data (depth) to the x channel in world coordinates.
    points_3d[..., 1] = x_coords  # Assign x camera data (left) to the y channel in world coordinates.
    points_3d[..., 2] = y_coords  # Assign y camera data (up) to the z channel in world coordinates.  
    
    # If no extrinsic matrix is provided, return the points in camera coordinates
    if extrinsic is None:
        return points_3d

    # If an extrinsic matrix is provided, apply it to transform points to the world coordinates
    # Convert the points to homogeneous coordinates (by adding a 1 at the end of each [x, y, z] tuple)
    ones = torch.ones((height, width), device=device)
    # Use world coordinate system, x forward, z up, y left
    points_homogeneous = torch.stack([z_coords, x_coords, y_coords, ones], dim=-1)  # (height, width, 4)
    
    # Apply the extrinsic transformation (4x4 matrix) to convert points to the world frame
    points_transformed = torch.matmul(points_homogeneous, extrinsic.T)  # Multiply by the transpose of extrinsic
    
    # Drop the homogeneous coordinate (the last dimension) and return the first 3 coordinates (x, y, z)
    return points_transformed[..., :3]


def filter_invalid_points(points, device=None):
    """
    Filters out invalid points: NaN values or points at the origin.

    Args:
        points (torch.Tensor): Tensor of shape [num_points, 3 + num_features] containing 3D points and additional features.
        device (torch.device, optional): Device to run the computation on. Defaults to the device of the input tensor.

    Returns:
        tuple: If valid points are found, returns:
            - points (torch.Tensor): Tensor of shape [num_valid_points, 3 + num_features] containing filtered points with only valid ones remaining.
        If no valid points are found, returns None.
    """

    device = points.device if device is None else device  # Use specified device or infer from input

    # Ensure the tensor is on the correct device and contiguous in memory
    points = points.to(device).contiguous()

    # Extract the x, y, z coordinates from the points
    points_coordinates = points[..., :3]

    # Mask: Check if all coordinates (x, y, z) are valid (i.e., not NaN)
    valid_points_mask = torch.all(points_coordinates == points_coordinates, dim=-1)

    # Mask: Check if points are not at the origin (0, 0, 0)
    is_not_origin_point = torch.any(points_coordinates != 0.0, dim=-1)
    valid_points_mask = torch.bitwise_and(valid_points_mask, is_not_origin_point)

    # Return None if no valid points are found
    if valid_points_mask.sum() == 0:
        return None

    # Select only the valid points (using the valid mask)
    points = points[valid_points_mask]

    return points


def get_cube_coordinates(points_coordinates, distances, warping_method):
    """
    Computes the cube side and 2D coordinates on the cube's surface for the given 3D points.

    Args:
        points_coordinates (torch.Tensor): Tensor of shape [num_points, 3] containing 3D coordinates of the points.
        distances (torch.Tensor): Tensor of shape [num_points] containing distances of the points from the origin.
        warping_method (callable): A function to unwarp UV coordinates for more homogeneous mapping on the sphere surface.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Indices of the cube sides the points are projected onto, shape [num_points].
            - torch.Tensor: 2D coordinates of the points on the cube sides, shape [num_points, 2].
    """
    
    device = points_coordinates.device

    # Normalize the 3D coordinates by their distances (get direction of points)
    normalized_point_coordinates = points_coordinates / distances.view(-1, 1)

    # Find the dimension where the absolute value is maximal to determine the cube side
    max_dim_idxs = torch.argmax(torch.abs(normalized_point_coordinates), dim=-1)
    max_dim_signs = torch.sign(
        torch.gather(normalized_point_coordinates, dim=-1, index=max_dim_idxs.view(-1, 1))
    ).flatten()

    # Compute the side index based on the dominant dimension and its sign
    cube_face_idx = torch.zeros_like(max_dim_idxs)
    cube_face_idx = torch.where((max_dim_idxs == 0) & (max_dim_signs == -1), torch.tensor(1, device=device), cube_face_idx)
    cube_face_idx = torch.where((max_dim_idxs == 1) & (max_dim_signs == 1), torch.tensor(2, device=device), cube_face_idx)
    cube_face_idx = torch.where((max_dim_idxs == 1) & (max_dim_signs == -1), torch.tensor(3, device=device), cube_face_idx)
    cube_face_idx = torch.where((max_dim_idxs == 2) & (max_dim_signs == 1), torch.tensor(4, device=device), cube_face_idx)
    cube_face_idx = torch.where((max_dim_idxs == 2) & (max_dim_signs == -1), torch.tensor(5, device=device), cube_face_idx)

    # Concatenate normalized coordinates with their negatives for both positive and negative faces
    positive_and_negative_normalized_points = torch.cat([
        normalized_point_coordinates,
        -normalized_point_coordinates
    ], dim=1)

    # Predefined face mappings for each cube side (width, height, and cosine dimensions), used to select
    # which dimension (x, y, z, -x, -y, -z) of the original points to use for the width, height, and cosine
    # of the 2D coordinates on the cube sides.
    face_mapping = torch.tensor([
        [4, 5, 0],  # front
        [1, 5, 3],  # rear
        [0, 5, 1],  # left
        [3, 5, 4],  # right
        [4, 0, 2],  # top
        [4, 3, 5],  # bottom
    ], device=device)

    # Gather width, height, and cosine dimension indices for the points
    dim_idx_width_height_cosine = face_mapping[cube_face_idx]

    # Gather the corresponding normalized coordinates for the selected dimensions
    normalized_side_coordinates_w_h_c = torch.gather(
        positive_and_negative_normalized_points, dim=1, index=dim_idx_width_height_cosine
    )

    # Compute the UV coordinates (2D projection on the cube sides)
    uv_coordinates = normalized_side_coordinates_w_h_c[:, 0:2] / normalized_side_coordinates_w_h_c[:, [2]]

    # Apply the warping method to get the final 2D coordinates
    cube_face_coordinates = warping_method.unwarp(uv_coordinates)

    return cube_face_idx, cube_face_coordinates


def filter_and_project_onto_cube_sides(points=None, warping_method=None):
    """
    Projects 3D points onto the sides of a cube after filtering out invalid points.

    Args:
        points (torch.Tensor): Tensor of points with shape [num_points, 3 + num_features].
        warping_method (callable, optional): A function for warping points onto the cube sides.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Filtered points with shape [num_valid_points, 3 + num_features].
            - torch.Tensor: Distances of the valid points from the origin, shape [num_valid_points].
            - torch.Tensor: Indices of the cube sides the points are projected onto, shape [num_valid_points].
            - torch.Tensor: 2D coordinates of the points on the cube sides, shape [num_valid_points, 2].
    """

    # Filter points
    filtered_points = filter_invalid_points(points)

    if filtered_points is None:
        raise ValueError("No valid points found after filtering.")

    # Extract the 3D coordinates from the points (first 3 columns)
    points_coordinates = filtered_points[..., :3]

    # Compute the distance of each point from the origin
    distances = torch.norm(points_coordinates, dim=-1)

    # Compute cube side indices and 2D coordinates on the cube's surface
    cube_face_idx, cube_face_coordinates = get_cube_coordinates(
        points_coordinates=points_coordinates,
        distances=distances,
        warping_method=warping_method
    )

    return filtered_points, distances, cube_face_idx, cube_face_coordinates


def normalized_image_coordinates_to_image_coordinates(normalized_w, normalized_h, resolution):
    """
    Converts normalized floating point coordinates to quantized integer image coordinates.

    Args:
        normalized_w (torch.Tensor): Normalized width coordinates in the range [-1, 1].
        normalized_h (torch.Tensor): Normalized height coordinates in the range [-1, 1].
        resolution (int): The resolution of the image.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Quantized integer width coordinates in the range [0, resolution-1].
            - torch.Tensor: Quantized integer height coordinates in the range [0, resolution-1].
    """
    eps = 1e-5
    float_idx_w_floor = (normalized_w + 1.0) / 2.0 * (resolution - eps)
    float_idx_h_floor = (normalized_h + 1.0) / 2.0 * (resolution - eps)
    return torch.floor(float_idx_w_floor).type(torch.int), torch.floor(float_idx_h_floor).type(torch.int)


def scatter_nd(indices, updates, shape, reduction='sum', previous_values=None):
    output = torch.zeros(*shape, device=updates.device)
    if previous_values is not None:
        output = previous_values.clone()
    if reduction == 'sum':
        output.index_add_(0, indices.long().view(-1), updates.view(-1, updates.shape[-1]))
    else:
        raise NotImplementedError(f"Reduction '{reduction}' is not implemented.")
    return output


def project_features_to_sphere(cube_face_idx, cube_face_coordinates, distances, semantics, resolution, device):
    """
    Projects features from cube faces onto a spherical surface.

    Args:
        cube_face_idx (torch.Tensor): Indices of the cube sides the points are projected onto, shape [num_points].
        cube_face_coordinates (torch.Tensor): 2D coordinates of the points on the cube sides, shape [num_points, 2].
        distances (torch.Tensor): Distances of the points to be projected from the origin, shape [num_points].
        semantics (torch.Tensor): Semantics corresponding to each point that need to be projected, must be int64.
                                        shape [num_points].
        resolution (int): The resolution of the image on each cube face (height and width).
        initial_values (torch.Tensor): Initial values for the scatter operation, typically the starting value for the 
                                        result tensor.

    Returns:
        torch.Tensor: The projected features in the shape of a spherical surface, represented by 6 cube faces.
    """
    # check if semantics is an integer tensor
    if semantics.dtype != torch.int64:
        raise ValueError("Semantics should be an integer tensor.")
    if semantics.dim() == 1:
        semantics = semantics.unsqueeze(-1)
    if distances.dim() == 1:
        distances = distances.unsqueeze(-1)

    # Transform the normalized cube face coordinates (-1,1) to image coordinates (0, resolution-1)
    idx_w, idx_h = normalized_image_coordinates_to_image_coordinates(
        cube_face_coordinates[:, 0],  # u coordinates
        cube_face_coordinates[:, 1],  # v coordinates
        resolution             # Resolution of the cube face
    )

    # Prepare indices for the de-duping operation.
    idx = torch.stack([cube_face_idx, idx_w, idx_h], dim=1).long() # shape (num_points, 3)

    distance_significant_figures = 5 # No need to differentiate between distances less than 1e-5 meters.
    semantics_significant_figures = 3 # Likely less than 1000 unique semantics
    compound_key = ((distances * 10**distance_significant_figures).to(int) * 10**semantics_significant_figures) + semantics

    unique_keys, inverse_indices = torch.unique(idx, dim=0, return_inverse=True)
    # Get the minimum distance for each unique index using scatter_reduce
    deduped_compound_keys = torch.zeros(unique_keys.shape[0], 1, device=device, dtype=torch.int64)
    deduped_compound_keys = deduped_compound_keys.scatter_reduce(0, inverse_indices.unsqueeze(-1), compound_key, "min", include_self=False)

    # Unpack the compound key to get the distance and semantics
    deduped_semantics = (deduped_compound_keys % 10**semantics_significant_figures).to(torch.float32)
    deduped_distances = (deduped_compound_keys // 10**semantics_significant_figures).to(torch.float32) / 10**distance_significant_figures


    all_features = torch.cat([deduped_distances, deduped_semantics], dim=-1)
    result = torch.zeros(6, resolution, resolution, 2, device=device)
    result[unique_keys[:, 0], unique_keys[:, 1], unique_keys[:, 2], :] = all_features

    return result