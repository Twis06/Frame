import rospy
import json

import numpy as np
import torch

import time

import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

import cv2 

from rendering.ray_casting import Example
from rendering.mesh_utils import construct_window_outer_corners, convert_plane_window_to_trimesh

model_path = "/home/intel/policy_deploy_ws/src/px4ctrl/scripts/models/actor34200000.pt"
data_path = "/home/intel/policy_deploy_ws/src/px4ctrl/scripts/sa_pair.json"

class HILNode:
    def __init__(self):
        rospy.init_node('HIL_node')

        self.student = rospy.get_param('student', True)
        self.load_model(model_path)

        # Load the gap geometry
        self.gap_inner_vertices = np.array(rospy.get_param('gap_geometry/inner_vertices'), dtype=np.float32)
        # self.gap_outer_vertices = np.array(rospy.get_param('gap_geometry/outer_vertices'), dtype=np.float32)
        self.gap_outer_vertices = construct_window_outer_corners(self.gap_inner_vertices)
        rendered_vertices, rendered_triangles = convert_plane_window_to_trimesh(self.gap_outer_vertices, self.gap_inner_vertices)

        # Load the gap pose
        self.gap_pos = np.array(rospy.get_param('gap_pose/position'), dtype=np.float32)
        self.gap_ori = np.array(rospy.get_param('gap_pose/orientation'), dtype=np.float32) / 180.0 * np.pi

        # Construct gap points
        if not self.student:
            gap_keypoints = self.find_equidistant_points(self.gap_inner_vertices[:, 1:].copy(), 32)
            zeros = np.zeros((gap_keypoints.shape[0], 1))
            gap_keypoints = np.hstack((zeros, gap_keypoints))
            # print(gap_keypoints)
            self.gap_keypoints = self.transform_gate_keypoints(gap_keypoints, self.gap_ori[0], self.gap_ori[1], self.gap_ori[2], self.gap_pos)
            # print(self.gap_keypoints)

        # Init static robot states and actions variables 
        # observations
        self.shape = tuple(rospy.get_param('pixel/shape', [1, 256, 256]))  # H x W, removed batch dimension
        self.binary = np.array(rospy.get_param('pixel/binary', True), dtype=bool)
        self.use_velocity = np.array(rospy.get_param('use_velocity', True), dtype=bool)
        self.img_obs = torch.zeros(self.shape, dtype=torch.float32, device='cuda')  # Removed batch dimension
        self.vec_obs = torch.zeros(6, dtype=torch.float32, device='cuda') \
            if self.use_velocity else torch.zeros(2, dtype=torch.float32, device='cuda')
        # velocity+rpy or py
        self.last_cmd = torch.zeros(4, dtype=torch.float32, device='cuda')

        # states for virtual image rendering
        self.p = np.array([-4., 0., 1.], dtype=np.float32)
        self.q = np.array([1., 0., 0., 0.], dtype=np.float32)

        # Temporary states
        self.euler_cpu = torch.zeros(3, dtype=torch.float32, device='cpu')
        self.euler_gpu = torch.zeros(3, dtype=torch.float32, device='cuda')
        self.act_cpu = torch.zeros(4, dtype=torch.float32, device='cpu')
        self.act_gpu = torch.zeros(4, dtype=torch.float32, device='cuda')

        # Init rendering
        rendered_indices = rendered_triangles.flatten(order='C')
        self.camera = Example(rendered_vertices, rendered_indices, device='cuda', num_cameras=1)
        
        print("Gap geometry and pose loaded successfully!")
        with open(data_path, 'r') as f:
            self.data = json.load(f)
        
        self.start_warmup_time = None
        self.warmup_done = False

    def load_model(self, model_path):
        try:
            self.model = torch.jit.load(model_path)
            self.model.to('cuda')
            self.model.eval()

            if self.student:
                self.hidden_state = torch.zeros((1, 1, 256)).to('cuda')   
            
            print('\033[32m[px4ctrl] <HIL node> Model loaded successfully!\033[0m')
        except torch.jit.Error as e:
            rospy.logerr(f'[px4ctrl] <HIL node> Error loading the model: {str(e)}')
    
    def find_equidistant_points(self, vertices: np.ndarray, n: int) -> np.ndarray:
        assert n > 0, "Number of points must be positive"
        assert vertices.shape[1] == 2, "Each vertex must be a 2D point"
        
        vertex_count = vertices.shape[0]
        
        # Calculate perimeter
        edges = np.roll(vertices, -1, axis=0) - vertices
        edge_lengths = np.linalg.norm(edges, axis=1)
        perimeter = np.sum(edge_lengths)
        
        # Calculate segment length
        segment_length = perimeter / n
        
        # Initialize result with first vertex
        result = [vertices[0]]
        
        if n > 1:
            current_distance = 0.0
            current_vertex = 0
            
            # Find remaining points
            for i in range(1, n):
                target_distance = i * segment_length
                
                while current_distance < target_distance:
                    next_vertex = (current_vertex + 1) % vertex_count
                    edge_length = edge_lengths[current_vertex]
                    
                    if current_distance + edge_length < target_distance:
                        current_distance += edge_length
                        current_vertex = next_vertex
                    else:
                        ratio = (target_distance - current_distance) / edge_length
                        p = vertices[current_vertex] + ratio * edges[current_vertex]
                        result.append(p)
                        break
        
        return np.array(result)
    
    def transform_gate_keypoints(self, points: np.ndarray, roll: float, pitch: float, yaw: float, 
                           translation: np.ndarray = np.array([0, 0, 1.5])) -> np.ndarray:
        assert points.shape[1] == 3, "Points should have shape (num_points, 3)"
    
        points = points.T
        
        # Get rotation matrix
        rotation = rpy_to_rotation_matrix(roll, pitch, yaw)
        print(rotation)
        
        # Apply rotation
        transformed_points = rotation @ points
        
        # Apply translation
        transformed_points = transformed_points + translation.reshape(3, 1)

        transformed_points = transformed_points.T
        
        return transformed_points

    @torch.no_grad()
    def inference(self):
        if self.use_velocity:
            self.v_gpu.copy_(self.v_cpu)
        self.euler_gpu.copy_(self.euler_cpu)

        if self.use_velocity:
            self.vec_obs[:3] = self.v_gpu
            self.vec_obs[3:6] = self.euler_gpu
        else:
            self.vec_obs[:2] = self.euler_gpu[:2]
        
        # Add batch dimension before passing to model for both image and vector observations
        img_obs_batched = self.img_obs.unsqueeze(0)
        vec_obs_batched = self.vec_obs.unsqueeze(0)

        # Run the model
        self.act_gpu, self.hidden_state = self.model(img_obs_batched, vec_obs_batched, self.hidden_state.detach()) # TODO: check the shape
        self.act_gpu = torch.clamp(self.act_gpu, -1.0, 1.0)

    def render_virtual_image(self):
        equiv_quat, render_pos = self.camera.obj_equiv_pose(
            self.q, 
            self.p, 
            self.gap_ori, 
            self.gap_pos, 
            w_first=True
        )
        render_quat = self.camera.trans_body2cam(equiv_quat)
        
        self.camera.update_cam_pose(render_pos, render_quat)
        self.camera.render()
        img_with_batch = self.camera.get_binary_img_tensor() \
            if self.binary else self.camera.get_depth_img_tensor()
        
        # Remove batch dimension from rendered image
        self.img_obs = img_with_batch.squeeze(0)

        img_np = self.img_obs.cpu().squeeze(0).numpy()
        visualize_map(img_np, f"Pixel Obs - Camera 0", self.binary)

    def get_gap_points(self) -> np.ndarray:
        num_points = self.gap_keypoints.shape[0]
        
        rot_matrix = quaternion_to_rotation_matrix(self.q)
        rot_transpose = rot_matrix.T

        p_matrix = np.tile(self.p.reshape(3, 1), (1, num_points))
        
        gate_keypoints = self.gap_keypoints.copy().T
        
        transformed_points = rot_transpose @ (gate_keypoints - p_matrix)
        
        gap_points = transformed_points.flatten('F') 
        
        return gap_points

    def run(self):
        print("Starting warmup...")
        for _ in range(20):
            quatnp2eulertensor(self.q, self.euler_cpu)
            self.render_virtual_image()

            self.inference()
            self.hidden_state = torch.zeros((1, 1, 256)).to('cuda') if self.student else None
            self.act_cpu.copy_(self.act_gpu)

        print("Warmup done!")
 
        for step_index, step in enumerate(self.data):
            pixel = torch.tensor(step['obs']['img'], dtype=torch.float32)
            state = torch.tensor(step['obs']['state'], dtype=torch.float32)
            position = np.array(step['position'], dtype=np.float32)
            quat = np.array(step['quat'], dtype=np.float32)
            action_label = torch.tensor(step['action'], dtype=torch.float32)

            pixel = pixel.to('cuda')
            state = state.to('cuda')

            self.p = position[0]
            self.q = quat[0]
            quatnp2eulertensor(self.q, self.euler_cpu)

            self.render_virtual_image()

            self.inference()
            
            self.act_cpu.copy_(self.act_gpu)
            print("Action: ", self.act_cpu)

            error_pixel = torch.norm(self.img_obs - pixel)
            error_action = torch.norm(self.act_cpu - action_label)

            print(f"Step {step_index}: Pixel error = {error_pixel}, Action error = {error_action}")


def quatnp2eulertensor(quat: np.ndarray, euler: torch.Tensor):
    w = quat[0]
    x = quat[1]
    y = quat[2]
    z = quat[3]

    # roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    euler[0] = torch.atan2(torch.tensor(sinr_cosp), torch.tensor(cosr_cosp))

    # pitch (y-axis rotation)
    sinp = 2 * (w * y - z * x)

    assert -1.0 <= sinp <= 1.0, "invalid input for asin in pitch calculation"
    euler[1] = torch.asin(torch.tensor(sinp))

    # yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    euler[2] = torch.atan2(torch.tensor(siny_cosp), torch.tensor(cosy_cosp))

def visualize_map(image, title, binary):
    img_normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    img_normalized = img_normalized.astype(np.uint8)
    
    if not binary:
        img_normalized = cv2.applyColorMap(img_normalized, cv2.COLORMAP_VIRIDIS)
    
    cv2.imshow(title, img_normalized)
    time.sleep(0.01)
    cv2.waitKey(1)  

def world2body_velocity_matrix(odom_q: np.ndarray, world_vel: torch.Tensor):
    w, x, y, z = odom_q

    R = torch.tensor([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]
    ], dtype=world_vel.dtype, device=world_vel.device)
    
    R_trans = R.T
    return R_trans @ world_vel

def rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    # Roll (X-axis rotation)
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll), np.cos(roll)]
    ])
    
    # Pitch (Y-axis rotation)
    Ry = np.array([
        [np.cos(pitch), 0, np.sin(pitch)],
        [0, 1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    # Yaw (Z-axis rotation)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw), np.cos(yaw), 0],
        [0, 0, 1]
    ])
    
    # Combined rotation matrix: R = Rz * Ry * Rx
    return Rz @ Ry @ Rx
    
def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [    2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z,     2*y*z - 2*w*x],
        [    2*x*z - 2*w*y,     2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y]
    ])

if __name__ == '__main__':
    hil_node = HILNode()
    # if not hil_node.onboard:
    #     hil_node.wait_odom()
    hil_node.run()

    # Test the gap points
    # hil_node.q = np.array([9.999606e-01, -7.176600e-03,  5.221164e-03,  3.747168e-05], dtype=np.float32)
    # hil_node.p = np.array([-4.0355587, -1.5798399e-3, 1.5599259], dtype=np.float32)
    # gap_points = hil_node.get_gap_points()
    # print(gap_points)