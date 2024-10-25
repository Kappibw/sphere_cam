import matplotlib.pyplot as plt
import numpy as np

def visualize_point_cloud(points_3d, depth_image, downsample_factor=10):
    """
    Visualizes the 3D point cloud and the original depth image side by side.

    Args:
        points_3d (torch.Tensor): A tensor of shape (height, width, 3) representing the 3D coordinates (x, y, z).
        depth_image (torch.Tensor): A 2D tensor representing the depth values for each pixel.
        downsample_factor (int): Factor by which to downsample the 3D point cloud. Higher numbers will result in fewer points.
    """
    # Downsample the 3D points by selecting every n-th point based on the downsample_factor
    downsampled_points_3d = points_3d[::downsample_factor, ::downsample_factor]
    
    # Convert the downsampled tensor to a numpy array for visualization
    points_np = downsampled_points_3d.cpu().numpy()
    
    # Extract x, y, z coordinates
    x = points_np[..., 0].flatten()
    y = points_np[..., 1].flatten()
    z = points_np[..., 2].flatten()
    
    # Create a figure for both the depth image and 3D point cloud
    fig = plt.figure(figsize=(12, 6))
    
    # Plot the depth image (2D)
    ax1 = fig.add_subplot(121)
    ax1.imshow(depth_image.cpu().numpy(), cmap='viridis')
    ax1.set_title('Original Depth Image')
    ax1.set_xlabel('Width')
    ax1.set_ylabel('Height')
    plt.colorbar(ax1.imshow(depth_image.cpu().numpy(), cmap='viridis'), ax=ax1)

    # Create the 3D point cloud plot
    ax2 = fig.add_subplot(122, projection='3d')
    
    # Plot the points in 3D
    ax2.scatter(x, y, z, c=z, cmap='viridis', marker='o', s=1, alpha=0.8)
    
    # Set labels and title for the 3D plot
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    ax2.set_title('Downsampled 3D Point Cloud Visualization')

    # Set the camera view: look from (0, 0, -5) towards the origin
    ax2.view_init(elev=90, azim=90)
    
    plt.tight_layout()
    plt.show()


def visualize_cube_sphere(cube_tensor):
    """
    Visualize a cube-sphere tensor with depth and semantic data for each face.
    
    Args:
        cube_tensor (torch.Tensor): A tensor of shape (6, 64, 64, 2), where the last dimension is 
                                    (depth, semantic_class).
    """
    # Verify tensor shape
    if cube_tensor.shape != (6, 64, 64, 2):
        raise ValueError("Expected tensor shape to be (6, 64, 64, 2).")
    
    # Set up the layout of faces for the cube representation
    # We will assume the order of faces is: front, left, right, back, top, bottom
    face_layout = [
        [None,    4,    None,  None],  # top
        [1,       0,    2,     3],     # middle row (left, front, right, back)
        [None,    5,    None,  None]   # bottom
    ]

    # Prepare figure for visualization
    fig, axes = plt.subplots(3, 4, figsize=(16, 8))
    plt.suptitle("Cube Sphere Visualization - Depth and Semantic Classes")

    # Display the depth and semantic data separately
    for row in range(3):
        for col in range(4):
            # Find which face corresponds to this grid cell
            face_index = face_layout[row][col]
            
            if face_index is None:
                # No face for this slot; hide this subplot
                axes[row, col].axis('off')
            else:
                # Display depth data
                depth_img = cube_tensor[face_index, :, :, 0].cpu().numpy()
                axes[row, col].imshow(depth_img, cmap='viridis')
                axes[row, col].set_title(f"Face {face_index} - Depth")
                axes[row, col].axis('off')
    
    # Create a new figure for semantic data
    fig, axes = plt.subplots(3, 4, figsize=(16, 8))
    plt.suptitle("Cube Sphere Visualization - Semantic Classes")

    for row in range(3):
        for col in range(4):
            face_index = face_layout[row][col]
            if face_index is None:
                axes[row, col].axis('off')
            else:
                # Display semantic class data
                semantic_img = cube_tensor[face_index, :, :, 1].cpu().numpy()
                axes[row, col].imshow(semantic_img, cmap='tab20')
                axes[row, col].set_title(f"Face {face_index} - Semantic")
                axes[row, col].axis('off')
    plt.show()
