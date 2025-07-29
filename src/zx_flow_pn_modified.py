import sys
import os
import numpy as np
import math
import quaternion
import time
from datetime import datetime
import random
# import torch
from itertools import product

from scipy.spatial import ConvexHull

# from torch.utils.tensorboard import SummaryWriter
# from ruamel.yaml import YAML

# from utils import quat2direction, euler2quat, quat2mat, load_model

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CURRENT_DIR)
ROOT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(ROOT_DIR)

# from algorithms.ppo import PPO
# from algorithms.model import LoopCNN1DFusion, PointMLPFusion

import rospy
from sensor_msgs.msg import Imu
from tf.transformations import euler_from_quaternion
from nav_msgs.msg import Odometry
from mavros_msgs.msg import AttitudeTarget, OpticalFlowRad, RCIn
from tf.transformations import quaternion_from_euler
# from nlink_parser.msg import TofsenseMFrame0
import csv
from scipy.spatial.transform import Rotation
import threading

# distillation_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../distillation'))
# sys.path.append(distillation_dir_path)
# from distillation.pixel_env_wrapper.rendering.ray_casting import Example
# from distillation.pixel_env_wrapper.rendering.mesh_utils import assemble_meshes

class BufferSearcher:
    def __init__(self, max_length=20):
        self.msg_buffer = []
        self.max_length = max_length

    def store_msg(self, msg):
        self.msg_buffer.append(msg)
        if len(self.msg_buffer) > self.max_length:
            self.msg_buffer.pop(0)

    def get_closest_msg(self, target_time):
        closest_msg = None
        min_diff = float('inf')
        for m in self.msg_buffer:
            diff = abs(m.header.stamp.to_sec() - target_time.to_sec())
            if diff < min_diff:
                closest_msg = m
                min_diff = diff
        return closest_msg

class CallCounter:
    def __init__(self):
        self.call_log = {} # {identifier: [time, current_count, last_count]}

    def calls_in_last_second(self, identifier: str, des_freq, flag_print = True) -> None:
        now = time.time()
        if identifier not in self.call_log:
            self.call_log[identifier] = [now, 1, 1]
        if now - self.call_log[identifier][0] > 1:
            cur_cnt = self.call_log[identifier][1]
            if flag_print:
                print(f"{identifier}: {cur_cnt} calls")
            if cur_cnt < des_freq * 0.95 or cur_cnt > des_freq * 1.1:
                print(f"\033[93mWarning: {identifier} frequency {cur_cnt} mismatch {des_freq}\033[0m")
            self.call_log[identifier] = [now, 1, cur_cnt]
        else:
            self.call_log[identifier][1] += 1
        
        return self.call_log[identifier][2]

class EKFDrone:
    def __init__(self):
        # System parameters
        self.g = 9.81                     # Gravitational acceleration

        # Noise matrices
        self.acc_gyro_noise = [2.0, 2.0, 2.0, 0.1, 0.1, 0.1] # from px4 log
        self.R_vel_bxy = np.diag(np.power([0.5, 0.5], 2)) # Body-frame xy velocity observation noise (see /drone/odom_flow_raw)
        self.R_h = np.diag(np.power([1.0], 2)) # Height observation noise (see /drone/odom_flow_raw). A very big value is expected here.
        self.R_vel_wz = np.diag(np.power([0.6], 2)) # World-frame z velocity observation noise (see /drone/odom_flow_raw)
        
        # Initialize state
        self.dt = 0.1                     # Time step (updated in predict)
        self.state_dim = 9 # [x, y, z, vx, vy, vz, phi, theta, psi]
        self.state = np.zeros(self.state_dim)
        self.cov = np.eye(self.state_dim) * 0.1
        self.Q = np.eye(self.state_dim) * 0.1 # Do not change, Q is updated automatically

    def rotation_matrix(self, phi, theta, psi):
        """Calculate rotation matrix from body to world coordinates (ZYX order)"""
        # for scipy version > 1.4.0
        # return Rotation.from_euler('ZYX', [psi, theta, phi]).as_matrix()
        # for scipy version < 1.4.0
        return Rotation.from_euler('ZYX', [psi, theta, phi]).as_dcm()

    def euler_derivative_matrix(self, phi, theta):
        """Euler angle derivative transformation matrix (with angle clamping)"""
        theta = np.clip(theta, -np.pi/2 + 1e-6, np.pi/2 - 1e-6)
        return np.array([
            [1, np.sin(phi)*np.tan(theta), np.cos(phi)*np.tan(theta)],
            [0, np.cos(phi), -np.sin(phi)],
            [0, np.sin(phi)/np.cos(theta), np.cos(phi)/np.cos(theta)]
        ])

    def compute_rp_from_acc(self, ax, ay, az):
        """Compute roll, pitch from accelerometer data only"""
        roll = np.arctan2(ay, az) # Roll (phi) calculation
        pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2)) # Pitch (theta) calculation
        norm = np.linalg.norm([ax, ay, az])
        return roll, pitch, norm

    def state_transition(self, x, u):
        """Nonlinear state transition function"""
        x_new = x.copy()
        
        # Unpack state
        vx, vy, vz = x[3], x[4], x[5]
        phi, theta, psi = x[6], x[7], x[8]
        
        # Unpack input
        a_bx, a_by, a_bz, p, q, r = u
        
        # Position update
        x_new[0] += vx * self.dt
        x_new[1] += vy * self.dt
        x_new[2] += vz * self.dt
        
        # Velocity update (adjust gravity term)
        R = self.rotation_matrix(phi, theta, psi)
        acc_world = R @ np.array([a_bx, a_by, a_bz]) - np.array([0, 0, self.g])
        x_new[3:6] += acc_world * self.dt
        # print("time=", time.time(), "x_new[3:6]: ", x_new[3:6].T, " acc_world: ", acc_world.T, " dt: ", self.dt, " dx=", acc_world.T * self.dt)
        
        # Attitude update (with angle clamping)
        T = self.euler_derivative_matrix(phi, theta)
        euler_rates = T @ np.array([p, q, r])
        # print("phi, theta, psi: ", phi, theta, psi, " p, q, r: ", p, q, r)
        x_new[6:9] += euler_rates * self.dt
        # print("time=", time.time(), "x_new[6:9]: ", x_new[6:9].T, " euler_rates: ", euler_rates.T, " dt: ", self.dt, " dx=", euler_rates.T * self.dt)

        return x_new

    def compute_jacobian_F(self, x, u):
        """Compute state transition Jacobian matrix"""
        F = np.eye(self.state_dim)
        
        # Unpack state and input
        phi, theta, psi = x[6], x[7], x[8]
        a_bx, a_by, a_bz, p, q, r = u
        
        # Position partial derivatives with respect to velocity
        F[0:3, 3:6] = np.eye(3) * self.dt
        
        # Velocity partial derivatives with respect to attitude angles
        R = self.rotation_matrix(phi, theta, psi)
        a_b = np.array([a_bx, a_by, a_bz])
        
        # Compute Jacobian using perturbation method
        delta = 1e-6
        for i, angle in enumerate([phi, theta, psi]):
            d_angle = np.zeros_like(x)
            d_angle[6+i] = delta
            
            R_perturbed = self.rotation_matrix(*(x[6:9] + d_angle[6:9]))
            dR = (R_perturbed @ a_b - R @ a_b) / delta
            F[3:6, 6+i] = dR * self.dt
        
        # Attitude partial derivatives with respect to attitude angles
        T = self.euler_derivative_matrix(phi, theta)
        angular_rates = np.array([p, q, r])
        J_att = np.eye(3)
        J_att += self.dt * np.array([
            [q*np.cos(phi)*np.tan(theta) - r*np.sin(phi)*np.tan(theta),
            (q*np.sin(phi) + r*np.cos(phi))/np.cos(theta)**2,
            0],
            [-q*np.sin(phi) - r*np.cos(phi),
            0,
            0],
            [(q*np.cos(phi) - r*np.sin(phi))/np.cos(theta),
            (q*np.sin(phi) + r*np.cos(phi))*np.sin(theta)/np.cos(theta)**2,
            0]
        ])
        F[6:9, 6:9] = J_att
        
        return F

    def observation_model_vel_b_xy(self, x):
        """Observation model for velocity: returns [body_vx, body_vy]"""
        phi, theta, psi = x[6], x[7], x[8]
        R = self.rotation_matrix(phi, theta, psi).T
        body_vel = R @ x[3:6]  # World velocity to body velocity
        return np.array([body_vel[0], body_vel[1]])

    def compute_jacobian_H_vel_b_xy(self, x):
        """Compute observation Jacobian matrix for velocity"""
        phi, theta, psi = x[6], x[7], x[8]
        R = self.rotation_matrix(phi, theta, psi).T
        
        # Partial derivatives with respect to velocity (correct dimensions)
        H_vel = R[0:2, :]  # First two rows, all three columns (2x3)
        
        # Partial derivatives with respect to attitude angles (using perturbation method)
        delta = 1e-6
        H_att = np.zeros((2, 3))
        for i in range(3):
            d_angle = np.zeros(3)
            d_angle[i] = delta
            R_perturbed = self.rotation_matrix(*(x[6:9] + d_angle)).T
            body_vel_perturbed = R_perturbed @ x[3:6]
            H_att[:, i] = (
                np.array([body_vel_perturbed[0], body_vel_perturbed[1]]) -
                self.observation_model_vel_b_xy(x)
            ) / delta
        
        # Combine Jacobian matrix
        H = np.zeros((2, self.state_dim))
        H[:, 3:6] = H_vel
        H[:, 6:9] = H_att
        
        return H

    def observation_model_vel_w_z(self, x):
        """Observation model for z velocity in world frame: returns [vz]"""
        return np.array([x[5]])

    def compute_jacobian_H_vel_w_z(self, x):
        """Compute observation Jacobian matrix for z velocity in world frame"""
        H = np.zeros((1, self.state_dim))
        H[0, 5] = 1.0
        return H

    def observation_model_height(self, x):
        """Observation model for height: returns [world height z]"""
        return np.array([x[2]])

    def compute_jacobian_H_height(self, x):
        """Compute observation Jacobian matrix for height"""
        H = np.zeros((1, self.state_dim))
        H[0, 2] = 1.0  # Height partial derivative with respect to z
        return H

    def get_state(self):
        """Get the current state in terms of position, velocity, and orientation"""
        pos = self.state[:3]
        vel = self.state[3:6]
        orientation = self.state[6:9]
        return pos, vel, orientation, self.cov

    def predict(self, dt, u):
        """Prediction step
           u: [ax, ay, az, rx, ry, rz]"""
        self.dt = dt
        F = self.compute_jacobian_F(self.state, u)
        self.state = self.state_transition(self.state, u)
        for i in range(6):  # Process noise covariance
            self.Q[3 + i, 3 + i] = self.acc_gyro_noise[i] * (self.dt ** 2)
        self.cov = F @ self.cov @ F.T + self.Q
        self.cov[:2, :2] = np.clip(self.cov[:2, :2], -10000, 10000) # pos x ang y are unobservable
        return self.state

    def update_vel_b_xy(self, vel_b_xy):
        """Update step for body-frame xy velocity"""
        H_vel = self.compute_jacobian_H_vel_b_xy(self.state)
        y_vel = vel_b_xy - self.observation_model_vel_b_xy(self.state)
        # print("A vel_b_xy=", vel_b_xy, " B vel_b_xy=", self.observation_model_vel_b_xy(self.state), " y_vel=", y_vel)
        S_vel = H_vel @ self.cov @ H_vel.T + self.R_vel_bxy
        S_vel_det = np.linalg.det(S_vel)
        inv_S_vel = np.array([
            [S_vel[1, 1], -S_vel[0, 1]],
            [-S_vel[1, 0], S_vel[0, 0]]
        ]) / S_vel_det
        K_vel = self.cov @ H_vel.T @ inv_S_vel
        # np.set_printoptions(linewidth=np.inf)
        # print("\n1",f"[{time.time()}]","self.cov=", self.cov, " H_vel=", H_vel, " inv_S_vel=", inv_S_vel)
        self.state[:8] += (K_vel @ y_vel)[:8] # avoid updating yaw as it is not observable [zxzx]
        # self.state += (K_vel @ y_vel)
        # np.set_printoptions(linewidth=np.inf)
        # print(f"[{time.time()}]","K_vel=", K_vel, "y_vel=", y_vel, " @=", K_vel @ y_vel)
        self.cov = (np.eye(self.state_dim) - K_vel @ H_vel) @ self.cov
        # np.set_printoptions(linewidth=np.inf)
        # print(f"[{time.time()}]","self.cov=", self.cov, " \n@=", (np.eye(self.state_dim) - K_vel @ H_vel))
        return self.state

    def update_vel_w_z(self, vel_w_z):
        """Update step for z velocity in world frame"""
        H_velz = self.compute_jacobian_H_vel_w_z(self.state)
        y_velz = vel_w_z - self.observation_model_vel_w_z(self.state)
        S_velz = H_velz @ self.cov @ H_velz.T + self.R_vel_wz
        K_velz = (self.cov @ H_velz.T) / S_velz
        # np.set_printoptions(linewidth=np.inf)
        # print("\n3",f"[{time.time()}]","self.cov=", self.cov, " H_velz=", H_velz, " S_velz=", S_velz)
        self.state[:8] += (K_velz @ y_velz)[:8] # avoid updating yaw as it is not observable [zxzx]
        # np.set_printoptions(linewidth=np.inf)
        # print(f"[{time.time()}]","K_velz=", K_velz, "y_velz=", y_velz, " @=", K_velz @ y_velz)
        self.cov = (np.eye(self.state_dim) - K_velz @ H_velz) @ self.cov
        # np.set_printoptions(linewidth=np.inf)
        # print(f"[{time.time()}]","self.cov=", self.cov, " \n@=", (np.eye(self.state_dim) - K_velz @ H_velz))
        return self.state

    def update_height(self, z_height):
        """Update step for height"""
        H_height = self.compute_jacobian_H_height(self.state)
        y_height = z_height - self.observation_model_height(self.state)
        S_height = H_height @ self.cov @ H_height.T + self.R_h
        K_height = self.cov @ H_height.T / S_height
        # np.set_printoptions(linewidth=np.inf)
        # print("\n2",f"[{time.time()}]","self.cov=", self.cov, " H_height=", H_height, " S_height=", S_height)
        self.state[:8] += (K_height @ y_height)[:8] # avoid updating yaw as it is not observable [zxzx]
        # np.set_printoptions(linewidth=np.inf)
        # print(f"[{time.time()}]","K_height=", K_height, "y_height=", y_height, " @=", K_height @ y_height)
        self.cov = (np.eye(self.state_dim) - K_height @ H_height) @ self.cov
        # np.set_printoptions(linewidth=np.inf)
        # print(f"[{time.time()}]","self.cov=", self.cov, " \n@=", (np.eye(self.state_dim) - K_height @ H_height))
        return self.state

class DroneIO:

    tofm_ready = False
    opt_flow_valid_time = 0
    mode = "MANUAL"
    mode_channel = 0
    imu_buffer = BufferSearcher()
    # kf = UAVKalmanFilter()
    kf = EKFDrone()
    cc = CallCounter()

    def __init__(self):
        self.lock = threading.Lock()  # Mutex for thread synchronization

        self.attitude_pub = rospy.Publisher("/mavros/setpoint_raw/attitude", AttitudeTarget, queue_size=10)
        self.odom_pub = rospy.Publisher("/drone/odom_propagate", Odometry, queue_size=10)
        self.flow_odom_pub = rospy.Publisher("/drone/odom_flow_raw", Odometry, queue_size=10)

        rospy.init_node("imu_euler_node")
        rospy.Subscriber("/mavros/imu/data", Imu, self.imu_callback, queue_size=2)
        # rospy.Subscriber("/vins_estimator/imu_propagate", Odometry, self.odometry_callback, queue_size=2)
        rospy.Subscriber("/mavros/px4flow/raw/optical_flow_rad", OpticalFlowRad, self.optical_flow_callback, queue_size=2)
        # rospy.Subscriber("/nlink_tofsensem_frame0", TofsenseMFrame0, self.tofm_callback, queue_size=2)
        rospy.Subscriber("/mavros/rc/in", RCIn, self.rc_in_callback, queue_size=2)

    def rc_in_callback(self, msg):
        rc_in_channels = msg.channels
        self.mode_channel = rc_in_channels[4]
        self.mode = "OFFBOARD" if rc_in_channels[4] > 1750 else "MANUAL"
        # rospy.loginfo("Received RC input channels: %s", self.rc_in_channels)

    def imu_callback(self, msg):
        with self.lock:
            t0 = time.time()
            self.imu_q = quaternion.quaternion(msg.orientation.w, msg.orientation.x, msg.orientation.y, msg.orientation.z)
            self.imu_buffer.store_msg(msg)
            if not hasattr(self, 'last_imu_time'):
                self.last_imu_time = msg.header.stamp
            if time.time() - self.opt_flow_valid_time < 1.0: # 1.0s timeout
                dt = (msg.header.stamp - self.last_imu_time).to_sec()
                self.kf.predict(dt, [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z, msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z])
            # self.cc.calls_in_last_second("imu_callback", 200, False)
            self.cc.calls_in_last_second("imu_callback", 200, False)

            if True:
                self.publish_odom(msg)

            # print(f"IMU callback time: {time.time() - t0:.4f}")
            self.last_imu_time = msg.header.stamp

    def odometry_callback(self, msg):
        self.odom_p = msg.pose.pose.position
        self.odom_q = quaternion.quaternion(msg.pose.pose.orientation.w, msg.pose.pose.orientation.x, msg.pose.pose.orientation.y, msg.pose.pose.orientation.z)
        self.odom_euler = np.array(euler_from_quaternion([self.odom_q.x, self.odom_q.y, self.odom_q.z, self.odom_q.w]))
        self.odom_v = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z])

    def optical_flow_callback(self, msg):

        if msg.distance < 0:
            return

        if not hasattr(self, "last_gyro") and np.isnan(msg.integrated_xgyro):
            return


        with self.lock:
            t0 = time.time()
            self.optical_flow = msg
            # flow_scale = 0.8
            flow_scale = 1.0
            max_dz = 3.0
            des_freq = 50.0
            # des_freq = 110.0
            v_scale = 1.22

            flow_quality_thresh = 40

            if msg.quality > flow_quality_thresh and msg.distance > 0:
                self.opt_flow_valid_time = time.time()
            # print("gyro1 = ",np.array([msg.integrated_xgyro, msg.integrated_ygyro]))
            # initialization
            if not hasattr(self, "last_gyro") and not np.isnan(msg.integrated_xgyro):
                self.last_gyro = self.last_last_gyro = np.array([msg.integrated_xgyro, msg.integrated_ygyro])
            if not hasattr(self, 'last_height_valid'):
                self.last_height_valid, self.last_height_valid_time, self.last_height = False, msg.header.stamp, msg.distance

            vel, height = np.array([0.0, 0.0, 0.0]), 0.0

            if msg.quality > flow_quality_thresh:
                flow = np.array([msg.integrated_x, msg.integrated_y])
                # print("flow = ", flow)
                gyro = self.last_last_gyro
                # gyro = (self.last_gyro + self.last_last_gyro) / 2.0

                # print("gyro = ", gyro)
                vel_xy = (flow * flow_scale - gyro) * des_freq * msg.distance * v_scale
                vel_xy = [-vel_xy[1], vel_xy[0]]
                # print("vel_xy = ", vel_xy)

                if not np.isnan(msg.integrated_xgyro):
                    self.last_last_gyro = self.last_gyro
                    self.last_gyro = np.array([msg.integrated_xgyro, msg.integrated_ygyro])
                self.kf.update_vel_b_xy(vel_xy)
                vel[:2] = vel_xy

            if msg.distance > 0:
                imu = self.imu_buffer.get_closest_msg(msg.header.stamp - rospy.Duration(0.05))
                x, y, z, w = imu.orientation.x, imu.orientation.y, imu.orientation.z, imu.orientation.w
                body_z3 = w*w - x*x - y*y + z*z
                height = self.ground_height_estimator(msg.distance * body_z3, 1/des_freq)
                self.kf.update_height(height)

                if self.last_height_valid:
                    dt = (msg.header.stamp - self.last_height_valid_time).to_sec()
                    dz = (height - self.last_height) * des_freq
                    dz = np.clip(dz, -max_dz, max_dz)
                    self.kf.update_vel_w_z(dz)
                    vel[2] = dz
                self.last_height_valid, self.last_height_valid_time, self.last_height = True, msg.header.stamp, height
            else:
                self.last_height_valid = False

            # For parameter tunning
            # if False:
            # if True:
            #     self.publish_odom(msg)
            #     if msg.quality > flow_quality_thresh and msg.distance > 0:
            #         self.publish_odom_flow(vel, height)
            #     self.cc.calls_in_last_second("optical_flow_callback", int(des_freq), False)

            ## Debugging
            # if not hasattr(self, 'last_state'):
            #     self.last_state = self.kf.state
            # if np.any(np.abs(self.kf.state[3:6] - self.last_state[3:6]) > 0.8):
            #     print(f"Velocity jump detected! Last state: {self.last_state}, Current state: {self.kf.state}")
            #     rospy.signal_shutdown("Velocity jump detected")
            #     sys.exit(0)
            # self.last_state = self.kf.state
            # print(f"Optical flow callback time: {time.time() - t0:.4f}")

    def tofm_callback(self, msg):
        self.tofm = msg
        if hasattr(self.tofm, 'pixels') and len(self.tofm.pixels) == 64 :
            self.tofm = np.array([pixel.dis / 1000.0 if pixel.dis_status == 0 else 4.0 for pixel in self.tofm.pixels])
            drone_io.tofm = drone_io.tofm.reshape(8, 8)[::-1, ::-1].flatten()
            # rospy.loginfo("TofM distances: %s", self.tofm)
        
        # self.tofm = np.full(64, 4.0)
        self.cc.calls_in_last_second("tofm_callback", 15, False)

        self.tofm_ready = True

    def publish_attitude(self, roll, pitch, yaw, thrust=0.5):
        q_w = quaternion_from_euler(*self.kf.get_state()[2])
        q_w = quaternion.quaternion(q_w[3], q_w[0], q_w[1], q_w[2])
        q = quaternion_from_euler(roll, pitch, yaw)
        q = quaternion.quaternion(q[3], q[0], q[1], q[2])
        q = self.imu_q * q_w.inverse() * q
        msg = AttitudeTarget()
        msg.type_mask = AttitudeTarget.IGNORE_ROLL_RATE | AttitudeTarget.IGNORE_PITCH_RATE | AttitudeTarget.IGNORE_YAW_RATE
        msg.orientation.x = q.x
        msg.orientation.y = q.y
        msg.orientation.z = q.z
        msg.orientation.w = q.w
        msg.thrust = thrust
        self.attitude_pub.publish(msg)

    def publish_odom(self,msg):
        odom_msg = Odometry()
        # odom_msg.header.stamp = rospy.Time.now() - rospy.Duration(0.1)
        # odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.stamp = msg.header.stamp

        # odom_msg.header.stamp = rospy.Time.now() - rospy.Duration(0.0001)

        odom_msg.header.frame_id = "world"
        odom_msg.child_frame_id = "body"

        pos, vel, orientation, cov = self.kf.get_state()

        print("pos = ",pos[0],"  ",pos[1],"  ",pos[2])

        odom_msg.pose.pose.position.x = pos[0]
        odom_msg.pose.pose.position.y = pos[1]
        odom_msg.pose.pose.position.z = pos[2]

        q = quaternion_from_euler(orientation[0], orientation[1], orientation[2])
        odom_msg.pose.pose.orientation.x = q[0]
        odom_msg.pose.pose.orientation.y = q[1]
        odom_msg.pose.pose.orientation.z = q[2]
        odom_msg.pose.pose.orientation.w = q[3]

        odom_msg.twist.twist.linear.x = vel[0]
        odom_msg.twist.twist.linear.y = vel[1]
        odom_msg.twist.twist.linear.z = vel[2]

        # Fill covariances (pose: x,y,z + roll,pitch,yaw; twist: linear + no angular data)
        pos_cov = cov[0:3, 0:3]
        ang_cov = cov[6:9, 6:9]
        vel_cov = cov[3:6, 3:6]

        for i in range(3):
            for j in range(3):
                # Pose: position
                odom_msg.pose.covariance[i * 6 + j] = pos_cov[i, j]
                # Pose: orientation
                odom_msg.pose.covariance[(i + 3) * 6 + (j + 3)] = ang_cov[i, j]
                # Twist: linear
                odom_msg.twist.covariance[i * 6 + j] = vel_cov[i, j]

        if hasattr(self, 'odom_pub'):
            self.odom_pub.publish(odom_msg)
        else:
            rospy.logerr("Odometry publisher not initialized.")

    def publish_odom_flow(self, vel, height):
        odom_msg = Odometry()
        odom_msg.header.stamp = rospy.Time.now()
        odom_msg.header.frame_id = "world"
        odom_msg.child_frame_id = "opt_flow"
        odom_msg.pose.pose.position.z = height
        odom_msg.twist.twist.linear.x = vel[0]
        odom_msg.twist.twist.linear.y = vel[1]
        odom_msg.twist.twist.linear.z = vel[2]
        

        self.flow_odom_pub.publish(odom_msg)

    def act2rpyt(self, act, dt):
        euler = self.kf.get_state()[2]
        if not hasattr(self, 'last_rpyt'):
            self.last_rpyt = np.array([*euler, 0.0])
        act_mean = np.array([0.0, 0.0, 0.0, 0.5])
        act_std = np.array([np.pi / 6.0, np.pi / 6.0, np.pi / 2.0, 0.5])  # roll, pitch, yaw rate, thrust signal

        rpyt = act * act_std + act_mean # roll, pitch, yaw rate, thrust signal
        if self.mode == "MANUAL":
            rpyt[2] = euler[2]
        else:
            rpyt[2] = self.last_rpyt[2] + rpyt[2] * dt
            rpyt[2] = (rpyt[2] + np.pi) % (2 * np.pi) - np.pi  # Keep yaw in the range [-π, π]
        self.last_rpyt = rpyt
        return rpyt

    def ground_height_estimator(self, height, dt):
        max_dz = 3.0 * dt # 3.0 m/s max velocity
        converge_thresh, converge_time = 0.3, 5.0 # 0.3m, 5.0s
        if not hasattr(self, 'last_h'):
            self.last_h, self.gh = height, 0.0
        
        if abs(height - self.last_h) > max_dz and abs(height - self.kf.get_state()[0][2]) > max_dz:
            self.gh += self.last_h - height

        conv_step = converge_thresh / converge_time * dt
        if abs(self.gh) < converge_thresh and abs(self.gh) > conv_step * 2:
            self.gh -= np.sign(self.gh) * conv_step

        self.last_h = height
        return height + self.gh

class PolicyMode():
    ACTOR = 1.0
    REF = 0.0
    RESET = -1.0

class GoalInfo():
    def __init__(self):
        # temporary gap info structure
        self.goal_pos = np.array([40, 0, 1.5], dtype=np.float32)
        self.gap_vertices_world = []

class InitConfig:
    def __init__(self):
        # prefab configuration
        # self.prefab_names = ["rectangle", "rhombus", "triangle"]
        # self.prefab_names = ["rectangle","triangle","ellipse","rhombus"]
        self.prefab_names = ["rectangle"]

        # position configuration
        self.x_box = (-29, -27)
        # self.x_box = (-20., -19.5)
        self.y_box = (-10, 10)
        self.z_box = (0.5, 2.5)
        assert self.x_box[0] <= self.x_box[1]
        assert self.y_box[0] <= self.y_box[1]
        assert self.z_box[0] <= self.z_box[1]
        self.x_resolution = 0.3
        self.y_resolution = 0.3
        self.z_resolution = 0.3
        # self.x_resolution = 1
        # self.y_resolution = 1
        # self.z_resolution = 1

        # pose configuration (for training only)
        # self.random_yaw = np.pi/4
        self.random_yaw = 0.

        # gap pose configuration 
        # self.roll_range = (0.0, np.pi/2.2)
        self.roll_range = (0.0, 0.0)
        self.pitch_range = (0.0, 0.0)
        self.angle_resolution = np.pi/14
        # self.angle_resolution = np.pi/10

def angle_between_body_and_world_z(q):

    # Rotate the local Z-axis [0, 0, 1] by the given quaternion
    x, y, z, w = q.x, q.y, q.z, q.w
    body_z_axis = np.array([
        2.0 * (x*z + w*y),
        2.0 * (y*z - w*x),
        w*w - x*x - y*y + z*z
    ], dtype=np.float32)

    # Calculate angle using the dot product
    dot_val = np.dot(body_z_axis, [0, 0, 1])
    dot_val = np.clip(dot_val, -1.0, 1.0)  # avoid numerical issues
    return np.arccos(dot_val)

def my_reward_func(info, goal_info):
    # 1. position-related feature: Distance to gap center
    pos_diff = info["pos"][:, :2] - goal_info.goal_pos[:2]
    last_pos_diff = info["last_pos"][:, :2] - goal_info.goal_pos[:2]
    distance_to_gap = np.linalg.norm(pos_diff, axis=1)
    last_distance_to_gap = np.linalg.norm(last_pos_diff, axis=1)
    
    distance_progress = last_distance_to_gap - distance_to_gap
    distance_progress = np.where(last_distance_to_gap - distance_to_gap < 4 * 0.033333, last_distance_to_gap - distance_to_gap, 4 * 0.033333)
    distance_progress = np.where(last_distance_to_gap - distance_to_gap > -4 * 0.033333, last_distance_to_gap - distance_to_gap, -4 * 0.033333)

    # 2. direction penalty
    x_direction = np.array([1, 0])
    pos_diff = info["pos"][:, :2] - info["last_pos"][:, :2]
    pos_direction = np.where(
        np.linalg.norm(pos_diff, axis=1, keepdims=True) > 0.01,
        pos_diff / np.linalg.norm(pos_diff, axis=1, keepdims=True),
        np.zeros_like(pos_diff)
    )
    alignment = np.dot(pos_direction, x_direction)
    direction_penalty = alignment - 1

    # 2. action magnitude penalty
    action_magnitude = np.abs(info["act"][:, 0]) + np.abs(info["act"][:, 1]) + 2 * np.abs(info["act"][:, 2]) + 2 * np.abs(info["act"][:, 3])
    action_magnitude_penalty = -action_magnitude

    # 3. action change penalty
    action_change_penalty = - np.where((info["steps"] > 1), np.linalg.norm(info["act"] - info["last_act"], axis=1), 0)

    # 4. max velocity penalty
    vel = np.linalg.norm(info["vel"], axis=1)
    max_vel_penalty = np.where(vel > 4.0, -np.exp(abs(vel - 4.0)) + 1, 0)
    max_vel_penalty = np.where(max_vel_penalty < -5, -5, max_vel_penalty)    # limit the velocity  penalty within 100
    min_vel_penalty = np.where(vel < 1.0, -np.exp(abs(1.0 - vel)) + 1, 0)   
    min_vel_penalty = np.where(min_vel_penalty < -5, -5, min_vel_penalty)    # limit the velocity  penalty within 100
    vel_penalty = max_vel_penalty + min_vel_penalty

    # 5. z position penalty
    z_pos = info["pos"][:, 2]
    floor_dist = z_pos - (-3)  
    floor_penalty = np.where(floor_dist < 1.0, -np.exp(2*(1.0 - floor_dist)) + 1, 0)
    ceiling_dist = 5 - z_pos  
    ceiling_penalty = np.where(ceiling_dist < 1.0, -np.exp(2*(1.0 - ceiling_dist)) + 1, 0)

    z_position_penalty = floor_penalty + ceiling_penalty

    # 6. collision penalty
    obstacle_collision_penalty = -1.0 * (info["collided"]) * vel 

    # 7. ESDF-guided reward
    # r = c * (1 - e^(-kd^2))
    k = 100
    d = info["closet_dist"]
    esdf_reward = 5 * (1 - np.exp(-k * d**2))
    # esdf_reward = 2 * (1 - np.exp(-k * d**2))
    esdf_reward += info["closet_dist"] - info["last_closet_dist"]

    # 8. succeed reward
    succeed_reward = info["succeed"]

    # Reward calculation
    reward_components = np.column_stack([
        distance_progress,  
        direction_penalty,
        action_magnitude_penalty,
        action_change_penalty,
        vel_penalty,
        z_position_penalty,
        obstacle_collision_penalty,
        esdf_reward,
        succeed_reward
    ])

    # coefficients for each reward component
    coeffs = np.array([3.0, 0.1, 0.00, 0.12, 0.5, 0.0, 20.0, 0.0, 20.0])
    # coeffs = np.array([3.0, 0.1, 0.05, 0.12, 0.5, 0.0, 20., 0.0, 20.0])  The initial value of 20250109
    # print("reward components: ", reward_components)
    # print("weighted reward components: ", coeffs * reward_components)
    
    # calculate total reward
    reward = np.sum(coeffs * reward_components, axis=1)
    
    # set reward to 0 for done episodes that haven't crossed
    # reward = np.where(info["done"] & ~info["crossed"], 0, reward)

    # print("reward: ", reward)
    
    return reward

class TrainConfig:
    def __init__(self):
        # training setup
        self.epoches = 2e3
        self.rollout_horizon = int(300)
        self.ref_rollout_horizon = int(0)
        self.eval_interval_epoch = 20
        self.eval_trials = 1
        self.training =  True
        self.evaluating = True
        self.saving_model = True

        # general algorithm parameters
        self.gamma = 0.99
        self.batch_size = int(15000) 
        self.learning_epoches = 5
        self.reward_scale = 2 ** 0
        self.learning_rate_actor = 1e-4
        self.learning_rate_critic = 1e-4
        self.clip_grad_norm = 3.
        self.action_max = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

        # PPO specific parameters
        self.ratio_clip = 0.20
        self.lambda_gae_adv = 0.95  
        self.lambda_entropy = 0.025
        self.target_kl = None

        # tricks
        self.state_value_tau = 0.1
        self.reward_scaling = False
        self.adv_norm = True
        self.adam_eps = 1e-5
        self.if_tanh_activation = False # ReLU converges faster than Tanh but is lass stable and may go down after the peak

        # network 
        # self.feature_extractor = PointMLPFusion
        self.feature_extractor = None
        self.shared_feature = False
        self.mlp_dims = [256, 256, 128]

        # training pipeline setup
        # self.prior_traj_path = ["/home/intel/Codes/reset_state_rectangle"]
        self.prior_traj_path = None
        self.negative_mining = False
        self.reset_map_skips = 3

        self.device = []


if __name__ == "__main__":

    # print("\033[91mself.tofm = np.full(64, 4.0)\033[0m")

    # hist_len = 5
    # vec_obs_dim = 13
    # pixel_obs_dim = 8*8
    # obs_dim = pixel_obs_dim + vec_obs_dim
        
    # 1. create online rl learner
    # train_config = TrainConfig()
    # train_config.device = "cuda:0"
    # # feature_extractor = train_config.feature_extractor(vec_obs_dim, num_cloud_points, embedding_dim=32, hidden_dim=128, if_tanh_activation=train_config.if_tanh_activation) \
    # #     if train_config.feature_extractor is not None else None
    # learner = PPO(1, train_config, vec_obs_dim=vec_obs_dim * hist_len, pixel_obs_dim=pixel_obs_dim * hist_len, act_dim=4, \
    #               device=torch.device(train_config.device), feature_extractor=None, shared_feature=train_config.shared_feature, if_tanh_activation=train_config.if_tanh_activation)  
    
    # # 2. load model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # load_model(learner, path = os.path.join(current_dir, f"/home/dji/workspace/DroneEnd2End/online_rl/model/epoches_T0_160.pt"), device=train_config.device)

    # # 4. train!
    # writer = SummaryWriter(comment=f"|mlp_dims=[128, 128, 64, 64]|amp=0|le=0.025|bs=15000|num_envs=4096|")
    
    drone_io = DroneIO()
    rospy.Rate(1).sleep()

    while not rospy.is_shutdown():
        missing_attrs = []
        # if not hasattr(drone_io, 'tofm'):
        #     missing_attrs.append('tofm')
        # if not hasattr(drone_io, 'odom_v'):
        #     missing_attrs.append('odom_v')
        # if not hasattr(drone_io, 'odom_euler'):
        #     missing_attrs.append('odom_euler')
        if not hasattr(drone_io, 'optical_flow'):
            missing_attrs.append('optical_flow')
        if missing_attrs:
            rospy.loginfo("Missing attributes: %s", ', '.join(missing_attrs))
        else:
            break
        rospy.Rate(1).sleep()  # 1 Hz

    csv_file_path = os.path.join(current_dir, "flight_data.csv")
    with open(csv_file_path, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["time", "mode_channel", "v_x", "v_y", "v_z", "euler_x", "euler_y", "euler_z", "last_act_roll", "last_act_pitch", "last_act_yaw_rate", "last_act_thrust", "p_z", "act_roll", "act_pitch", "act_yaw_rate", "act_thrust", "rpyt_roll", "rpyt_pitch", "rpyt_yaw", "rpyt_thrust"])

    print("Start running!")
    last_act = np.zeros(4)
    last_act_time = time.time()
    rate = rospy.Rate(1000)  # 1000 Hz
    # obs = np.zeros((1, obs_dim), dtype=np.float32)
    # obs_hist = np.zeros((hist_len, 1, obs_dim), dtype=np.float32)
    while not rospy.is_shutdown():
        # if drone_io.tofm_ready:
        #     # Perform actions when tof data is ready
        #     # rospy.loginfo("TofM data is ready.")
        #     cur_act_time = time.time()

        #     # obs = np.array([self.tofm])
        #     vel_ref = np.array([0.98, 0.2])
        #     # obs = np.concatenate([drone_io.tofm, drone_io.odom_v, drone_io.odom_euler, last_act, vel_ref, np.array([drone_io.odom_p.z]), last_act]) # add height as obs last_pos[:, 2][:, np.newaxis]
        #     # obs_once = np.concatenate([drone_io.tofm, drone_io.odom_v, drone_io.odom_euler, last_act, vel_ref, np.array([drone_io.odom_p.z])]).reshape(1, -1).astype(np.float32)
        #     # obs = np.concatenate([drone_io.tofm, drone_io.odom_v, drone_io.odom_euler, vel_ref, np.array([drone_io.odom_p.z]), last_act]).reshape(1, -1).astype(np.float32)
        #     pos, vel, rot, _ = drone_io.kf.get_state()
        #     obs = np.concatenate([drone_io.tofm, vel, rot, vel_ref, [pos[2]], last_act]).reshape(1, -1).astype(np.float32)
        #     if hist_len > 1:
        #         obs_hist[:-1] = obs_hist[1:]
        #     obs_hist[-1] = obs
        #     obs = np.concatenate([obs_hist[i] for i in range(hist_len)], axis=1)

        #     act = learner.get_action(obs, eval=True).reshape(-1)
        #     rpyt = drone_io.act2rpyt(act, cur_act_time - last_act_time)
        #     drone_io.publish_attitude(rpyt[0], rpyt[1], rpyt[2], rpyt[3])

        #     np.set_printoptions(linewidth=np.inf)
        #     rospy.loginfo(
        #         "\033[91mObs: tofm=\n%s\n\033[0m, \033[92mv=%s\033[0m, \033[93meuler=%s\033[0m, \033[94mlast_act=%s\033[0m, \033[95mvel_ref=%s\033[0m, \033[96mp.z=%s\033[0m \nAct: %s, RPYT: %s",
        #         np.array2string(drone_io.tofm.reshape(8, 8), separator=', '), vel, rot, last_act, vel_ref, pos[2], act, rpyt
        #     )

        #     with open(csv_file_path, 'a', newline='') as csvfile:
        #         writer = csv.writer(csvfile)
        #         row_data = [time.time()] \
        #                    + [drone_io.mode_channel] \
        #                    + list(vel) \
        #                    + list(rot) \
        #                    + list(last_act) \
        #                    + [pos[2]] \
        #                    + list(act) \
        #                    + list(rpyt)
        #         writer.writerow(row_data)

        #     last_act = act
        #     last_act_time = cur_act_time
        #     drone_io.tofm_ready = False  # Reset the flag after processing

        #     drone_state = drone_io.kf.get_state()
        #     # print(f"drone_state: P: {[f'{state:.3f}' for state in drone_state[0]]}, "
        #     #       f"\tV: {[f'{state:.3f}' for state in drone_state[1]]}, "
        #     #       f"\tR: {[f'{state:.3f}' for state in drone_state[2]]}")

        rate.sleep()

    print("\ndeploy.py exit")

    
