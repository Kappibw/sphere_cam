import torch
import sphere_cam
from pathlib import Path

file_path = Path(__file__).parent / "test_images/test_depth.pt"
depth_image = torch.load(file_path)

intrinsic = torch.tensor([[369.7771, 0.0, 489.9926], 
                          [0.0, 369.7771, 275.9385], 
                          [0.0, 0.0, 1.0]], dtype=torch.float32)

# Position and quaternion of camera from world (egosphere) origin
pos = torch.tensor([0.4761, 0.0035, 0.1055], dtype=torch.float32)
w, x, y, z = 0.9914449, 0.0, 0.1305262, 0.0

# Rotation matrix from quaternion
rotation_matrix = torch.tensor([
    [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]
], dtype=torch.float32)

# Create a 4x4 extrinsic matrix
extrinsic = torch.eye(4, dtype=torch.float32)
extrinsic[:3, :3] = rotation_matrix
extrinsic[:3, 3] = pos

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transfer depth image and matrices to the correct device
depth_image = depth_image.to(device)
intrinsic = intrinsic.to(device)
extrinsic = extrinsic.to(device)

# Generate the 3D point cloud
points_3d = sphere_cam.deproject_pointcloud(depth_image, intrinsic, extrinsic=extrinsic, device=device)

# Visualize the 3D point cloud
sphere_cam.visualize_point_cloud(points_3d, depth_image)

points, distances, cube_face_idx, cube_face_coordinates = sphere_cam.filter_and_project_onto_cube_sides(points_3d, sphere_cam.TangentialWarp())
# Create some dummy semantics
test_semantics = (distances - 0.5).to(int)
sphere_projection = sphere_cam.project_features_to_sphere(cube_face_idx, cube_face_coordinates, distances, semantics=test_semantics, resolution=64, device=device)
sphere_cam.visualize_cube_sphere(sphere_projection)