import warp as wp
import torch

import numpy as np
import time
from numpy.random import choice
from scipy import interpolate

import math
import numpy as np

import warp as wp
wp.config.verify_fp = True
wp.init()

@wp.struct
class RenderMesh:
    id: wp.uint64
    vertices: wp.array(dtype=wp.vec3)
    indices: wp.array(dtype=int)
    pos: wp.array(dtype=wp.vec3)
    rot: wp.array(dtype=wp.quat)

@wp.struct
class Cameras:
    horizontal: float
    vertical: float
    aspect: float
    tan: float
    pos: wp.array(dtype=wp.vec3)
    rot: wp.array(dtype=wp.quat)

class Example:
    def __init__(self, points, indices, device, num_cameras):
        self.device = device

        horizontal_aperture = 320
        vertical_aperture = 256
        aspect = vertical_aperture / horizontal_aperture

        self.width = 320
        self.height = 256
        self.num_pixels = self.width * self.height

        self.num_cameras = num_cameras

        points_ = points
        indices_ = indices

        self.x_axis = np.array([1.0, 0.0, 0.0])
        self.y_axis = np.array([0.0, 1.0, 0.0])
        self.z_axis = np.array([0.0, 0.0, 1.0])

        self.rot_x = quat_from_angle_axis(np.pi/2, np.array([1.0, 0.0, 0.0]))
        self.rot_z = quat_from_angle_axis(-np.pi/2, np.array([0.0, 0.0, 1.0]))
        self.quat_cam2body = quat_mul(self.rot_z, self.rot_x)

        with wp.ScopedDevice(device=self.device):
            # construct RenderMesh
            self.render_mesh = RenderMesh()
            self.mesh = wp.Mesh(
                points=wp.array(points, dtype=wp.vec3, requires_grad=False), indices=wp.array(indices, dtype=int)
            )
            self.render_mesh.id = self.mesh.id
            self.render_mesh.vertices = self.mesh.points
            self.render_mesh.indices = self.mesh.indices
            self.render_mesh.pos = wp.zeros(1, dtype=wp.vec3, requires_grad=False)
            self.render_mesh.rot = wp.array(np.array([0.0, 0.0, 0.0, 1.0]), dtype=wp.quat, requires_grad=False)
           
            # setup cameras
            self.cameras = Cameras()
            self.cameras.horizontal = horizontal_aperture
            self.cameras.vertical = vertical_aperture
            self.cameras.aspect = aspect * 1.04
            self.cameras.tan = np.tan(np.radians(82 / 2))
            self.cameras.pos = wp.array(np.zeros((self.num_cameras, 3)), dtype=wp.vec3, requires_grad=False)
            self.cameras.rot = wp.array(np.array([0.0, 0.0, 0.0, 1.0] * self.num_cameras).reshape((self.num_cameras, 4)), dtype=wp.quat, requires_grad=False)

            self.depths = wp.zeros((self.num_cameras * self.num_pixels), dtype=float, requires_grad=False)

    def render(self):
        with wp.ScopedDevice(self.device):
            wp.launch(
                kernel=Example.draw_kernel,
                dim=self.num_cameras * self.num_pixels,
                inputs=[
                    self.num_cameras,
                    self.render_mesh,
                    self.cameras,
                    self.width,
                    self.height,
                    self.depths,
                ]
            )

    @wp.kernel
    def draw_kernel(
        num_cameras: int,
        mesh: RenderMesh,
        cameras: Cameras,
        rays_width: int,
        rays_height: int,
        depths: wp.array(dtype=float),
    ):
        max_dist = 10.0

        tid = wp.tid()
        cam_id = tid // (rays_width * rays_height)
        pixel_id = tid % (rays_width * rays_height)

        x = pixel_id % rays_width
        y = rays_height - pixel_id // rays_width

        sx = 2.0 * float(x) / float(rays_width) - 1.0 
        sy = 2.0 * float(y) / float(rays_height) - 1.0

        # compute view ray in world space
        # offset_x = wp.float(cam2mesh[cam_id]) * 2. * max_dist  # use offset to determine which mesh to ray cast
        ro_world = cameras.pos[cam_id] 
        rd_cam = wp.vec3(sx * cameras.tan, sy * cameras.tan * cameras.aspect, -1.0)
        rd_body = wp.vec3(1.0, -sx * cameras.tan, sy * cameras.tan * cameras.aspect)
        rd_world = wp.normalize(wp.quat_rotate(cameras.rot[cam_id], wp.vec3(sx * cameras.tan, sy * cameras.tan * cameras.aspect, -1.0)))
        
        ry = math.atan(sy * cameras.tan * cameras.aspect)
        rx = math.atan(sx * cameras.tan)

        # compute view ray in mesh space
        inv = wp.transform_inverse(wp.transform(mesh.pos[0], mesh.rot[0]))
        ro = wp.transform_point(inv, ro_world)
        rd = wp.transform_vector(inv, rd_world)

        t = float(0.0)
        ur = float(0.0)
        vr = float(0.0)
        sign = float(0.0)
        n = wp.vec3()
        f = int(0)

        if wp.mesh_query_ray(mesh.id, ro, rd, max_dist + 0.1, t, ur, vr, sign, n, f):
            # dis_ = t * math.cos(ry) * math.cos(rx)  #! sometimes wrong but don't know why
            dis_ = t * wp.dot(wp.normalize(rd_cam), wp.vec3(0., 0., -1.))
        else:
            dis_ = max_dist

        depths[tid] = dis_

    def get_depth_vec_numpy(self):
        depth_numpy = self.depths.numpy()  # To read GPU array data back to CPU memory we can use array.numpy()
        # This will automatically synchronize with the GPU to ensure that any outstanding work has finished
        return depth_numpy
    
    def get_depth_img_tensor(self):
        depth_tensor = wp.torch.to_torch(self.depths.reshape((self.num_cameras, 1, self.height, self.width)))
        return depth_tensor
    
    def get_binary_img_tensor(self):
        depth_tensor = self.get_depth_img_tensor()
        binary_tensor = torch.where(depth_tensor < 10, torch.ones_like(depth_tensor), 
                              torch.zeros_like(depth_tensor))
        return binary_tensor
    
    def update_cam_pose(self, cam_pos: np.ndarray, cam_rot: np.ndarray, w_first=False):
        self.cameras.pos = wp.array(cam_pos, dtype=wp.vec3, requires_grad=False)
        self.cameras.rot = wp.array(cam_rot, dtype=wp.quat, requires_grad=False)

    def update_mesh_id(self, prefab_id: np.ndarray):
        self.cam2mesh = wp.array(prefab_id, dtype=int, requires_grad=False)
    
    def update_mesh_pose(self, mesh_pos: np.ndarray=None, mesh_rot: np.ndarray=None, w_first=False):
        if mesh_pos is not None:
            self.render_mesh.pos = wp.array(mesh_pos, dtype=wp.vec3, requires_grad=False)
        if mesh_rot is not None:
            self.render_mesh.rot = wp.array(mesh_rot, dtype=wp.quat, requires_grad=False)

    # def trans_body2cam(self, body_quat, w_first=True):
    #     if isinstance(body_quat, np.ndarray):
    #         quat = torch.tensor(body_quat, device="cpu")
    #     else:
    #         quat = body_quat.clone()
    #     if w_first: # transform from wxyz to xyzw
    #         quat = quat[:, [1, 2, 3, 0]]
    #     rot_x = quat_from_angle_axis(torch.tensor([np.pi/2], device="cpu"), torch.tensor([1.0, 0.0, 0.0], device="cpu"))[0]
    #     rot_z = quat_from_angle_axis(torch.tensor([-np.pi/2], device="cpu"), torch.tensor([0.0, 0.0, 1.0], device="cpu"))[0]
    #     quat_cam2body = quat_mul(rot_z, rot_x) 

    #     cam_quat = quat_mul(quat, quat_cam2body.repeat(self.num_cameras, 1))

    #     return cam_quat.numpy()
            
    def trans_body2cam(self, body_quat: np.ndarray, w_first: bool = True) -> np.ndarray:
        quat = body_quat.copy()
        if w_first:  # transform from wxyz to xyzw
            quat = quat[[1, 2, 3, 0]]
        
        cam_quats = np.zeros((self.num_cameras, 4))
        for i in range(self.num_cameras):
            cam_quats[i] = quat_mul(quat, self.quat_cam2body)
            
        return cam_quats
    
    # def obj_equiv_pose(self, body_quat, body_pos, obj_rpy, obj_pos, w_first=True):
    #     robot_quat = torch.tensor(body_quat, device="cpu", dtype=torch.float32) if isinstance(body_quat, np.ndarray) else body_quat.clone()
    #     robot_pos = torch.tensor(body_pos, device="cpu", dtype=torch.float32) if isinstance(body_pos, np.ndarray) else body_pos.clone()
    #     object_rpy = torch.tensor(obj_rpy, device="cpu", dtype=torch.float32) if isinstance(obj_rpy, np.ndarray) else obj_rpy.clone()
    #     object_pos = torch.tensor(obj_pos, device="cpu", dtype=torch.float32) if isinstance(obj_pos, np.ndarray) else obj_pos.clone()

    #     if w_first:
    #         robot_quat = robot_quat[:, [1, 2, 3, 0]]
        
    #     num_envs = robot_quat.shape[0]
        
    #     x_axis = torch.tensor([1.0, 0.0, 0.0], device="cpu").expand(num_envs, 3)
    #     y_axis = torch.tensor([0.0, 1.0, 0.0], device="cpu").expand(num_envs, 3)
    #     z_axis = torch.tensor([0.0, 0.0, 1.0], device="cpu").expand(num_envs, 3)
        
    #     quat_x = quat_from_angle_axis(object_rpy[:, 0], x_axis)  # Roll
    #     quat_y = quat_from_angle_axis(object_rpy[:, 1], y_axis)  # Pitch
    #     quat_z = quat_from_angle_axis(object_rpy[:, 2], z_axis)  # Yaw
        
    #     obj_quat = quat_mul(quat_z, quat_mul(quat_y, quat_x))
        
    #     obj_quat_inv = obj_quat.clone()
    #     obj_quat_inv[..., :3] *= -1

    #     equivalent_quat = quat_mul(obj_quat_inv, robot_quat)
    #     equivalent_quat = quat_unit(equivalent_quat)
        
    #     if object_pos.dim() == 1:
    #         object_pos = object_pos.expand(num_envs, 3)
        
    #     relative_pos = robot_pos - object_pos
        
    #     rotated_pos = quat_rotate(obj_quat_inv, relative_pos)
    #     equivalent_pos = rotated_pos

    #     equivalent_quat = equivalent_quat[:, [3, 0, 1, 2]]

    #     return equivalent_quat, equivalent_pos

    def obj_equiv_pose(self, 
                      body_quat: np.ndarray,
                      body_pos: np.ndarray, 
                      obj_rpy: np.ndarray,
                      obj_pos: np.ndarray,
                      w_first: bool = True):
        robot_quat = body_quat.copy()
        robot_pos = body_pos.copy()

        # print("robot_quat: ", robot_quat)
        
        if w_first:
            robot_quat = robot_quat[[1, 2, 3, 0]]
        
        quat_x = quat_from_angle_axis(obj_rpy[0], self.x_axis)  # Roll
        quat_y = quat_from_angle_axis(obj_rpy[1], self.y_axis)  # Pitch
        quat_z = quat_from_angle_axis(obj_rpy[2], self.z_axis)  # Yaw
        
        temp_quat = quat_mul(quat_y, quat_x)
        obj_quat = quat_mul(quat_z, temp_quat)
        
        obj_quat_inv = obj_quat.copy()
        obj_quat_inv[:3] *= -1

        equivalent_quat = quat_mul(obj_quat_inv, robot_quat)
        equivalent_quat = quat_unit(equivalent_quat)
        
        relative_pos = robot_pos - obj_pos
        equivalent_pos = quat_rotate(obj_quat_inv, relative_pos)
        
        equivalent_quat = equivalent_quat[[3, 0, 1, 2]]
        
        return equivalent_quat, equivalent_pos

# @torch.jit.script
# def quat_mul(a, b):
#     assert a.shape == b.shape
#     shape = a.shape
#     a = a.reshape(-1, 4)
#     b = b.reshape(-1, 4)

#     x1, y1, z1, w1 = a[:, 0], a[:, 1], a[:, 2], a[:, 3]
#     x2, y2, z2, w2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
#     ww = (z1 + x1) * (x2 + y2)
#     yy = (w1 - y1) * (w2 + z2)
#     zz = (w1 + y1) * (w2 - z2)
#     xx = ww + yy + zz
#     qq = 0.5 * (xx + (z1 - x1) * (x2 - y2))
#     w = qq - ww + (z1 - y1) * (y2 - z2)
#     x = qq - xx + (x1 + w1) * (x2 + w2)
#     y = qq - yy + (w1 - x1) * (y2 + z2)
#     z = qq - zz + (z1 + y1) * (w2 - x2)

#     quat = torch.stack([x, y, z, w], dim=-1).view(shape)

#     return quat
    
def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = a[0], a[1], a[2], a[3]
    x2, y2, z2, w2 = b[0], b[1], b[2], b[3]
    
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    ])

# def normalize(x, eps: float = 1e-9):
#     return x / x.norm(p=2, dim=-1).clamp(min=eps, max=None).unsqueeze(-1)

# def quat_unit(a):
#     return normalize(a)

def quat_unit(x: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    norm = np.maximum(np.linalg.norm(x), eps)
    return x / norm

def quat_from_angle_axis(angle: float, axis: np.ndarray) -> np.ndarray:
    theta = angle * 0.5
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    return np.concatenate([axis * sin_theta, [cos_theta]])

def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q_w = q[3]
    q_vec = q[:3]
    a = v * (2.0 * q_w * q_w - 1.0)
    b = np.cross(q_vec, v) * q_w * 2.0
    c = q_vec * np.dot(q_vec, v) * 2.0
    return a + b + c
