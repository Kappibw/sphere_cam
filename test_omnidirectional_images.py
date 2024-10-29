import torch
import sphere_cam
from pathlib import Path

def extrinsic_matrix_from_pos_quat(pos, quat):
    w, x, y, z = quat
    rotation_matrix = torch.tensor([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]
    ], dtype=torch.float32)
    extrinsic = torch.eye(4, dtype=torch.float32)
    extrinsic[:3, :3] = rotation_matrix
    extrinsic[:3, 3] = pos
    return extrinsic

file_path = Path(__file__).parent / "test_images/"

camera_data = {
    "forwards": {
        "file_path": file_path / "front_1.pt",
        "intrinsic": torch.tensor([[369.7771, 0.0, 489.9926],
                                   [0.0, 369.7771, 275.9385],
                                   [0.0, 0.0, 1.0]], dtype=torch.float32),
        "pos": torch.tensor([0.4761, 0.0035, 0.1055], dtype=torch.float32),
        "quaternion": torch.tensor([0.9914449, 0.0, 0.1305262, 0.0], dtype=torch.float32)
    },
    "backwards": {
        "file_path": file_path / "rear_1.pt",
        "intrinsic": torch.tensor([[369.7771, 0.0, 489.9926],
                                   [0.0, 369.7771, 275.9385],
                                   [0.0, 0.0, 1.0]], dtype=torch.float32),
        "pos": torch.tensor([-0.4641, 0.0035, 0.1055], dtype=torch.float32),
        "quaternion": torch.tensor([-0.001, 0.132, -0.005, 0.99], dtype=torch.float32)
    },
    "left": {
        "file_path": file_path / "left_1.pt",
        "intrinsic": torch.tensor([[369.7771, 0.0, 489.9926],
                                   [0.0, 369.7771, 275.9385],
                                   [0.0, 0.0, 1.0]], dtype=torch.float32),
        "pos": torch.tensor([0.0217, 0.1335, 0.1748], dtype=torch.float32),
        "quaternion": torch.tensor([0.6963642, -0.1227878, 0.1227878, 0.6963642], dtype=torch.float32)
    },
    "right": {
        "file_path": file_path / "right_1.pt",
        "intrinsic": torch.tensor([[369.7771, 0.0, 489.9926],
                                   [0.0, 369.7771, 275.9385],
                                   [0.0, 0.0, 1.0]], dtype=torch.float32),
        "pos": torch.tensor([0.0203, -0.1056, 0.1748], dtype=torch.float32),
        "quaternion": torch.tensor([0.6963642, 0.1227878, 0.1227878, -0.6963642], dtype=torch.float32)
    }
}

DOWNSAMPLE = 5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

all_points = []
for key, cam in camera_data.items():
    print(f"Processing {key} camera")
    depth_image = torch.load(cam["file_path"])
    extrinsic = extrinsic_matrix_from_pos_quat(cam["pos"], cam["quaternion"])
    intrinsic = cam["intrinsic"]
    intrinsic /= DOWNSAMPLE

    # Transfer depth image and matrices to the correct device
    depth_image = depth_image.to(device)
    intrinsic = intrinsic.to(device)
    # Transfer depth image and matrices to the correct device
    depth_image = depth_image.to(device)
    intrinsic = intrinsic.to(device)
    extrinsic = extrinsic.to(device)

    # Generate the 3D point cloud
    points_3d = sphere_cam.deproject_pointcloud(depth_image, intrinsic, extrinsic=extrinsic, device=device)

    # Visualize the 3D point cloud
    sphere_cam.visualize_point_cloud(points_3d, depth_image, cam["file_path"])

    all_points.append(points_3d)

points_3d = torch.cat(all_points, dim=0)
sphere_cam.visualize_point_cloud(points_3d, depth_image=None)

points, distances, cube_face_idx, cube_face_coordinates = sphere_cam.filter_and_project_onto_cube_sides(points_3d, sphere_cam.TangentialWarp())
# Create some dummy semantics
test_semantics = (distances - 0.5).to(int)
sphere_projection = sphere_cam.project_features_to_sphere(cube_face_idx, cube_face_coordinates, distances, semantics=test_semantics, resolution=64, device=device)
sphere_cam.visualize_cube_sphere(sphere_projection, camera_data)