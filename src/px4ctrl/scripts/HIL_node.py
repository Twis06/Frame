import rospy
from nav_msgs.msg import Odometry
from mavros_msgs.msg import AttitudeTarget
from mavros_msgs.msg import RCIn
from sensor_msgs.msg import Imu
from std_msgs.msg import Empty

import numpy as np
import torch

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Automatically initialize PyCUDA

import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

import time

from rendering.ray_casting import Example
from rendering.mesh_utils import construct_window_outer_corners, convert_plane_window_to_trimesh

import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class HILNode:
    def __init__(self):
        rospy.init_node('HIL_node')

        self.student = np.array(rospy.get_param('student', True), dtype=bool)

        # Load the model
        self.visualize_only = np.array(rospy.get_param('visualize_only', False), dtype=bool)
        if not self.visualize_only:
            self.if_tensorrt = rospy.get_param('use_tensorrt')
            model_path = rospy.get_param('model_path')
            self.load_model(model_path)
        
        # Load the gap geometry
        self.gap_inner_vertices = np.array(rospy.get_param('gap_geometry/inner_vertices'), dtype=np.float32)
        self.gap_outer_vertices = np.array(rospy.get_param('gap_geometry/outer_vertices'), dtype=np.float32)
        # self.gap_outer_vertices = construct_window_outer_corners(self.gap_inner_vertices)
        rendered_vertices, rendered_triangles = convert_plane_window_to_trimesh(self.gap_outer_vertices, self.gap_inner_vertices)

        # Load the gap pose
        self.gap_pos = np.array(rospy.get_param('gap_pose/position'), dtype=np.float32)
        self.gap_ori = np.array(rospy.get_param('gap_pose/orientation'), dtype=np.float32) / 180.0 * np.pi

        # Determine the number of gaps
        self.num_gaps = int(self.gap_pos.shape[0] / 3)
        self.cross_num = 0
        print(f"\033[32m[px4ctrl] <HIL node> There are {self.num_gaps} gaps to cross!\033[0m")

        # Construct gap points
        if not self.student:
            gap_keypoints_2d = self.find_equidistant_points(self.gap_inner_vertices[:, 1:].copy(), 32)
            zeros = np.zeros((gap_keypoints_2d.shape[0], 1))
            self.gap_keypoints_3d = np.hstack((zeros, gap_keypoints_2d))
            
            self.gap_keypoints = self.transform_gate_keypoints(self.gap_keypoints_3d, self.gap_ori[0], self.gap_ori[1], self.gap_ori[2], self.gap_pos[0:3])

        # Init static robot states and actions variables 
        # observations
        self.shape = tuple(rospy.get_param('pixel/shape', [1, 256, 256]))  # H x W, removed batch dimension
        self.cam_pos = np.array(rospy.get_param('pixel/extrinsic', False), dtype=bool)
        self.binary = np.array(rospy.get_param('pixel/binary', True), dtype=bool)
        self.use_velocity = np.array(rospy.get_param('use_velocity', True), dtype=bool)
        self.use_mocap_q = np.array(rospy.get_param('use_mocap_q', True), dtype=bool)
        if not self.use_mocap_q:
            self.q_imu = np.array([1., 0., 0., 0.], dtype=np.float32)
            self.roll_comp = rospy.get_param('roll_compensation', 0.0)
            self.pitch_comp = rospy.get_param('pitch_compensation', 0.0)
        self.img_obs = torch.zeros(self.shape, dtype=torch.float32, device='cuda')  # Removed batch dimension
        # state: velocity+rpy or rp
        if not self.if_tensorrt:
            self.vec_obs = torch.zeros(10, dtype=torch.float32, device='cuda') \
                if self.use_velocity else torch.zeros(6, dtype=torch.float32, device='cuda')
        else:
            self.vec_obs = torch.zeros(10, dtype=torch.float32, device='cpu') \
                if self.use_velocity else torch.zeros(6, dtype=torch.float32, device='cpu')
        
        self.last_cmd = torch.zeros(4, dtype=torch.float32, device='cuda') if not self.if_tensorrt else \
            torch.zeros(4, dtype=torch.float32, device='cpu')

        # states for virtual image rendering
        self.p = np.array([-4., 0., 1.], dtype=np.float32)
        self.q = np.array([1., 0., 0., 0.], dtype=np.float32)
        
        # Temporary states
        self.v_cpu = torch.zeros(3, dtype=torch.float32, device='cpu')
        self.v_gpu = torch.zeros(3, dtype=torch.float32, device='cuda')
        self.euler_cpu = torch.zeros(3, dtype=torch.float32, device='cpu')
        self.euler_gpu = torch.zeros(3, dtype=torch.float32, device='cuda')
        self.act_cpu = torch.zeros(4, dtype=torch.float32, device='cpu')
        self.act_gpu = torch.zeros(4, dtype=torch.float32, device='cuda')
        
        # Init rendering
        rendered_indices = rendered_triangles.flatten(order='C')
        self.camera = Example(rendered_vertices, rendered_indices, device='cuda', num_cameras=1)
        self.cam_pos = np.array(rospy.get_param('pixel/extristic', [0.0, 0.0, 0.0]), dtype=np.float32)
        
        self.visualize = np.array(rospy.get_param('pixel/visualize', False), dtype=bool)
        if self.visualize:
            self.bridge = CvBridge()
            self.image_pub = rospy.Publisher('/hil_node/virtual_image', Image, queue_size=1)

        # Stop trigger
        self.auto_stop = np.array(rospy.get_param('auto_stop', True), dtype=bool)
        if self.auto_stop:
            self.stop_pub = rospy.Publisher('/attitude_recovery', Empty, queue_size=10)
            self.stop_counter = 0

        # Callbacks
        self.onboard = np.array(rospy.get_param('onboard', False), dtype=bool)
        self.receive_odom = False
        self.last_odom_time = rospy.Time(0)
        self.last_receive_odom_time = rospy.Time(0)
        self.last_cmd_odom_time = rospy.Time(0)
        self.receive_rc_cmd = False  # rc ctrl
        self.check_inference_mode = 0.0
        self.have_init_last_check_inference_mode = False
        self.last_check_inference_mode = 0.0

        self.timeout_interval = 0
        if self.onboard:
            self.odom_sub = rospy.Subscriber('/ekf/ekf_odom', Odometry, self.odom_cb_onb, queue_size=1, tcp_nodelay=True)
        else:
            self.odom_sub = rospy.Subscriber('/ekf/ekf_odom', Odometry, self.odom_cb_ofb, queue_size=1, tcp_nodelay=True)
            self.start_warmup_time = None
            self.warmup_duration = np.array(rospy.get_param('warmup_duration', 8.0), dtype=float)
            self.warmup_done = False
            self.rc_in_sub = rospy.Subscriber('/mavros/rc/in', RCIn, self.rc_cmd_cb, queue_size=1, tcp_nodelay=True)
        if not self.use_mocap_q:
            self.imu_sub = rospy.Subscriber('/mavros/imu/data', Imu, self.imu_cb, queue_size=1, tcp_nodelay=True)

        self.ctrl_fcu_pub = rospy.Publisher('/hil_node/fcu_ctrl', AttitudeTarget, queue_size=1)
        self.cmd = AttitudeTarget()

        # Debug
        self.receive_odom_pub = rospy.Publisher('/hil_debug/receive_odom', Odometry, queue_size=1)

    def load_model(self, model_path):
        try:
            if not self.if_tensorrt:
                self.model = torch.jit.load(model_path)
                self.model.to('cuda')
                self.model.eval()
            else:
                print('[px4ctrl] <HIL node> Loading TensorRT engine...')
                with open(model_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
                    self.engine = runtime.deserialize_cuda_engine(f.read())
                
                self.inputs = []
                self.outputs = []
                self.bindings = []
                self.stream = cuda.Stream()

                for binding in self.engine:
                    size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
                    dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                    host_mem = cuda.pagelocked_empty(size, dtype)
                    device_mem = cuda.mem_alloc(host_mem.nbytes)
                    self.bindings.append(int(device_mem))
                    if self.engine.binding_is_input(binding):
                        self.inputs.append({'host': host_mem, 'device': device_mem})
                    else:
                        self.outputs.append({'host': host_mem, 'device': device_mem})

                self.context = self.engine.create_execution_context()

            if self.student:
                self.hidden_dim = rospy.get_param('hidden_dim')
                self.hidden_state = torch.zeros((1, 1, self.hidden_dim), dtype=torch.float32, device='cuda') if not self.if_tensorrt else \
                    torch.zeros((1, 1, self.hidden_dim), dtype=torch.float32, device="cpu")
            
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
        
        # Apply rotation
        transformed_points = rotation @ points
        
        # Apply translation
        transformed_points = transformed_points + translation.reshape(3, 1)

        transformed_points = transformed_points.T
        
        return transformed_points

    @torch.no_grad()
    def inference(self):
        if self.student:
            if self.use_velocity and not self.if_tensorrt:
                self.v_gpu.copy_(self.v_cpu)
            if not self.if_tensorrt:
                self.euler_gpu.copy_(self.euler_cpu)

            if self.use_velocity:
                self.vec_obs[:3] = self.v_gpu if not self.if_tensorrt else self.v_cpu
                self.vec_obs[3:6] = self.euler_gpu if not self.if_tensorrt else self.euler_cpu
                self.vec_obs[6:10] = self.last_cmd
            else:
                self.vec_obs[:6] = torch.concat([self.euler_gpu[:2], self.last_cmd], dim=0) if not self.if_tensorrt else self.euler_cpu[:2]
            
            # Add batch dimension before passing to model for both image and vector observations
            img_obs_batched = self.img_obs.unsqueeze(0)
            vec_obs_batched = self.vec_obs.unsqueeze(0)

            if self.if_tensorrt:
                cuda.memcpy_dtod_async(self.inputs[0]['device'], img_obs_batched.data_ptr(), img_obs_batched.element_size() * img_obs_batched.nelement(), self.stream)
                np.copyto(self.inputs[1]['host'], self.vec_obs.ravel())
                np.copyto(self.inputs[2]['host'], self.hidden_state.ravel())
                cuda.memcpy_htod_async(self.inputs[1]['device'], self.inputs[1]['host'], self.stream)
                cuda.memcpy_htod_async(self.inputs[2]['device'], self.inputs[2]['host'], self.stream)

            # Run the model
            if not self.if_tensorrt:
                # print("vec_obs now: ", self.vec_obs)
                self.act_gpu, self.hidden_state = self.model(img_obs_batched, vec_obs_batched, self.hidden_state.detach())  # TODO: check the shape
                self.act_gpu = torch.clamp(self.act_gpu, -1.0, 1.0)
            else:
                self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
                cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
                cuda.memcpy_dtoh_async(self.outputs[1]['host'], self.outputs[1]['device'], self.stream)
                self.hidden_state = self.outputs[0]['host']
                self.act_cpu = torch.tensor(self.outputs[1]['host']).clamp(-1.0, 1.0).detach()
        
        else:
            self.v_gpu.copy_(self.v_cpu)
            self.euler_gpu.copy_(self.euler_cpu)
            gap_points = torch.tensor(self.get_gap_points(), dtype=torch.float32, device='cuda')

            obs_teacher = torch.concat([gap_points, self.v_gpu, self.euler_gpu, self.last_cmd], dim=0)
            obs_teacher = obs_teacher.unsqueeze(0)

            self.act_gpu = self.model(obs_teacher).squeeze(0)
            self.act_gpu = torch.tanh(self.act_gpu)

    @torch.no_grad()
    def wait_odom(self):
        self.render_virtual_image()  # rendering use default states

        while not self.receive_odom:
            if not self.visualize_only:
                self.inference()
                self.model.reset_hidden_state()
                self.act_cpu.copy_(self.act_gpu)
            else:
                self.render_virtual_image()
                rospy.sleep(0.01)
        
        print(f'\033[32m[px4ctrl] <HIL node> Received odom (initial position: [{self.p[0]:.6f}, {self.p[1]:.6f}, {self.p[2]:.6f}]) and warmup done!\033[0m')

    def imu_cb(self, msg: Imu):
        self.q_imu[0] = msg.orientation.w
        self.q_imu[1] = msg.orientation.x
        self.q_imu[2] = msg.orientation.y
        self.q_imu[3] = msg.orientation.z

    def odom_cb_ofb(self, msg):  # offboard inference callback
        time_start_cb = time.perf_counter()
        
        # Check if the odom is stable
        if self.receive_odom:  
            if (msg.header.stamp - self.last_odom_time).to_sec() > 0.016:
                print(f"\033[33m[px4ctrl] <NIL node> The odom may be unstable! The last odom is sent {1000 * (msg.header.stamp - self.last_odom_time).to_sec():.4f}ms ago.\033[0m")
            if (rospy.Time.now() - self.last_receive_odom_time).to_sec() > 0.016:
                print(f"\033[33m[px4ctrl] <NIL node> The odom receive may be unstable! The last odom is received {1000 * (rospy.Time.now() - self.last_receive_odom_time).to_sec():.4f}ms ago.\033[0m")

        # Warmup 
        if not self.receive_odom:
            self.start_warmup_time = time.perf_counter()
        
        self.receive_odom = True
        self.last_odom_time = msg.header.stamp
        self.last_receive_odom_time = rospy.Time.now()

        # Check/measure odom transmission timeout
        # odom_pub_time = msg.header.stamp
        # odom_receive_time = rospy.Time.now()
        # time_diff = odom_receive_time - odom_pub_time
        # if (time_diff.to_sec() > 0.008 + 0.03):
        #     self.timeout_interval += 1
        #     rospy.logwarn(f"[px4ctrl] <NIL node> The odom is generated from {1000 * time_diff.to_sec():.4f}ms ago, which is too long! The timeout interval is {self.timeout_interval} now.")
        # else:
        #     # rospy.loginfo(f"[px4ctrl] <NIL node> The odom is generated from {1000 * time_diff.to_sec():.4f}ms ago.")
        #     self.timeout_interval = 0

        # Prepare the states info
        start_time = time.perf_counter()
        self.p[0] = msg.pose.pose.position.x
        self.p[1] = msg.pose.pose.position.y
        self.p[2] = msg.pose.pose.position.z
        self.q[0] = msg.pose.pose.orientation.w
        self.q[1] = msg.pose.pose.orientation.x
        self.q[2] = msg.pose.pose.orientation.y
        self.q[3] = msg.pose.pose.orientation.z
        self.v_cpu[0] = msg.twist.twist.linear.x
        self.v_cpu[1] = msg.twist.twist.linear.y
        self.v_cpu[2] = msg.twist.twist.linear.z
        self.v_cpu = world2body_velocity_matrix(self.q, self.v_cpu)
        if self.use_mocap_q:
            quatnp2eulertensor(self.q, self.euler_cpu)
        else:
            quatnp2eulertensor(self.q_imu, self.euler_cpu)
            self.euler_cpu[0] += self.roll_comp
            self.euler_cpu[1] += self.pitch_comp
        # end_time = time.perf_counter()
        # print("node 1: ", end_time-start_time)

        # Rendering
        # start_time = time.perf_counter()
        if self.student:
            self.render_virtual_image()
        # end_time = time.perf_counter()
        # print("node 2: ", 1000 * (end_time-start_time),  "ms")
        
        # Inference
        # start2 = time.perf_counter()
        if (not self.auto_stop and self.p[0] < self.gap_pos[-3] + 0.2) or (self.auto_stop and self.stop_counter < 10):
            if not self.visualize_only:
                if not self.warmup_done or not self.receive_rc_cmd:  
                    self.inference()
                    if self.student:  # reset hidden state before triggered
                        self.hidden_state = torch.zeros((1, 1, self.hidden_dim), dtype=torch.float32, device='cuda') if not self.if_tensorrt else \
                            torch.zeros((1, 1, self.hidden_dim), dtype=torch.float32, device='cpu')
                    if not self.if_tensorrt:
                        self.act_cpu.copy_(self.act_gpu)
                    # print("Warmup action: ",self.act_cpu)
                    
                    if not self.warmup_done and self.receive_rc_cmd:
                        rospy.logwarn(f"[px4ctrl] <HIL node> The warmup has not been done yet. Please switch back to the RC mode!")
                        self.receive_rc_cmd = False

                else:  # triggered to run the policy and control the vehicle
                    if self.cross_num < self.num_gaps - 1:
                        if not self.student:
                            if self.p[0] > self.gap_pos[3 * self.cross_num] + 0.15: # switch observation for consecutive gaps
                                self.gap_keypoints = self.transform_gate_keypoints(self.gap_keypoints_3d, self.gap_ori[3 * self.cross_num], self.gap_ori[3 * self.cross_num + 1], self.gap_ori[3 * self.cross_num + 2], 
                                                                                self.gap_pos[3 * self.cross_num : 3 * self.cross_num + 3])
                        else:
                            if self.p[0] > self.gap_pos[3 * self.cross_num] - 0.5:
                                if self.img_obs.sum(dim=(0,1,2)) == 0:
                                    self.cross_num += 1
                                    self.render_virtual_image()
                        
                    if (msg.header.stamp - self.last_cmd_odom_time).to_sec() > 0.014:  # 60~65Hz
                        self.last_cmd_odom_time = msg.header.stamp
                        self.receive_odom_pub.publish(msg)
                        
                        start_time = time.perf_counter()
                        self.inference()
                        end_time = time.perf_counter()
                        print("Inference Time: ", 1000 * (end_time-start_time), "ms")

                        print("Euler GPU: ", self.euler_gpu)

                        # Publish control commands
                        start_time = time.perf_counter()
                        if not self.if_tensorrt:
                            self.act_cpu.copy_(self.act_gpu)
                        
                        # print("Action: ", self.act_cpu)
                        
                        self.act_cpu = self.denormalize_action(self.act_cpu)  # TODO: shape change?
                        self.cmd.thrust = self.act_cpu[0][0].item()
                        self.cmd.body_rate.x = self.act_cpu[0][1].item()
                        self.cmd.body_rate.y = self.act_cpu[0][2].item()
                        self.cmd.body_rate.z = self.act_cpu[0][3].item()

                        # print("Cmd: ", self.cmd.thrust, ", ", self.cmd.body_rate.x)
                        
                        # Update last command for next iteration
                        if not self.if_tensorrt:
                            self.last_cmd.copy_(self.act_gpu)
                        else:
                            self.last_cmd.copy_(self.act_gpu.cpu())
                        # end_time = time.perf_counter()
                        # print("node 4: ", end_time-start_time)

                        # Publish control commands
                        self.ctrl_fcu_pub.publish(self.cmd)

                        if self.img_obs.sum(dim=(0,1,2)) == 0:  # determine if the vehicle has passed the gap
                            self.stop_counter += 1

        elif self.auto_stop and self.stop_counter >= 10 and self.warmup_done and self.receive_rc_cmd:
            self.stop_pub.publish(Empty())
            print("\033[32m[px4ctrl] <HIL node> Switch to blind control!\033[0m")

        # end2 = time.perf_counter()
        # print("node 3: ", 1000 * (end2 - start2), "ms")
        
        time_end_cb = time.perf_counter()
        if (time_end_cb - time_start_cb) > 0.014:
            print(f"\033[33m[px4ctrl] <HIL node> The main callback takes {1000 * (time_end_cb - time_start_cb):.4f}ms to run when warmup done is {self.warmup_done} trigger is {self.receive_rc_cmd}, which is too long!\n\033[0m")

        if time.perf_counter() - self.start_warmup_time > self.warmup_duration and not self.warmup_done:
            print(f"\033[32m[px4ctrl] <HIL node> Warmup done!\033[0m")
            self.warmup_done = True

    def odom_cb_onb(self, msg):  # onboard inference callback
        raise NotImplementedError("Onboard mode callback not implemented yet!")
    
    def rc_cmd_cb(self, msg):  
        self.check_inference_mode = (msg.channels[4] - 1000.0) / 1000.0  # 10 channel
        if not self.have_init_last_check_inference_mode:
            self.have_init_last_check_inference_mode = True
            self.last_check_inference_mode = self.check_inference_mode
        if self.last_check_inference_mode < 0.75 and self.check_inference_mode > 0.75 and not self.receive_rc_cmd:
            self.receive_rc_cmd = True
            print("\033[32m[px4ctrl] <HIL node> Receive RC Channel 10!!!\033[0m")
        elif self.check_inference_mode < 0.75:
            self.receive_rc_cmd = False

    def render_virtual_image(self):
        cam_pos_world = self.camera_to_world()
        equiv_quat, render_pos = self.camera.obj_equiv_pose(
            self.q, 
            self.p, 
            # cam_pos_world,
            self.gap_ori[self.cross_num * 3: self.cross_num * 3 + 3], 
            self.gap_pos[self.cross_num * 3: self.cross_num * 3 + 3], 
            w_first=True
        )
        render_quat = self.camera.trans_body2cam(equiv_quat)
        
        self.camera.update_cam_pose(render_pos, render_quat)
        self.camera.render()
        img_with_batch = self.camera.get_binary_img_tensor() \
            if self.binary else self.camera.get_depth_img_tensor()
        
        # Remove batch dimension from rendered image
        self.img_obs = img_with_batch.squeeze(0)
        
        if self.visualize:
            visualize_map(self.img_obs, "Virtual Image", self.binary, self.bridge, self.image_pub)

    def camera_to_world(self):
        w, x, y, z = self.q
        
        t = 2.0 * np.cross(self.q[1:], self.cam_pos)
        rotated_pos = self.cam_pos + w * t + np.cross(self.q[1:], t)
        
        return rotated_pos + self.p
    
    def get_gap_points(self) -> np.ndarray:
        num_points = self.gap_keypoints.shape[0]
        
        rot_matrix = quaternion_to_rotation_matrix(self.q)
        rot_transpose = rot_matrix.T

        p_matrix = np.tile(self.p.reshape(3, 1), (1, num_points))
        
        gate_keypoints = self.gap_keypoints.copy().T
        
        transformed_points = rot_transpose @ (gate_keypoints - p_matrix)
        
        gap_points = transformed_points.flatten('F') 
        
        return gap_points

    def denormalize_action(self, action: torch.Tensor):
        return action * ACT_STD + ACT_MEAN
    
ACT_MEAN = torch.tensor([[13., 0., 0., 0.]], dtype=torch.float32, device='cpu')
ACT_STD = torch.tensor([[7., 8., 2., 1.]], dtype=torch.float32, device='cpu')
    
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

def visualize_map(image, title, binary, bridge, image_pub):
    if torch.is_tensor(image):
        image_np = image.detach().cpu().numpy()
    else:
        image_np = np.array(image)

    # coords = np.where(image_np == 1)
    # if len(coords[0]) > 0:
    #     print("Found ones at coordinates (row, col):")
    #     for row, col in zip(coords[0], coords[1]):
    #         print(f"({row}, {col})")

    img_normalized = cv2.normalize(image_np, None, 0, 255, cv2.NORM_MINMAX)
    img_normalized = img_normalized.astype(np.uint8)

    if binary:
        if len(img_normalized.shape) == 3:
            img_normalized = img_normalized.squeeze()
        img_normalized = cv2.cvtColor(img_normalized, cv2.COLOR_GRAY2BGR)
    else:
        img_normalized = cv2.applyColorMap(img_normalized, cv2.COLORMAP_VIRIDIS)
    
    try:
        msg = bridge.cv2_to_imgmsg(img_normalized, "bgr8")
        msg.header.stamp = rospy.Time.now()
        image_pub.publish(msg)
    except CvBridgeError as e:
        rospy.logerr(f"[px4ctrl] <HIL node> Error converting image: {str(e)}")

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
    rospy.spin()
