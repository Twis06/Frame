import argparse
import symforce
symforce.set_epsilon_to_symbol()
import symforce.symbolic as sf
from symforce import typing as T
from symforce import ops
from symforce.values import Values
from derivation_utils import *

# The state vector is organized in an ordered dictionary
State = Values(
    quat_nominal = sf.Rot3(),
    vel = sf.V3(),
    pos = sf.V3(),
    gyro_bias = sf.V3(),
    accel_bias = sf.V3(),
    # mag_I = sf.V3(),
    # mag_B = sf.V3(),
    # wind_vel = sf.V2(),
    # terrain = sf.V1()
)

class VState(sf.Matrix):
    SHAPE = (State.storage_dim(), 1)

class VTangent(sf.Matrix):
    SHAPE = (State.tangent_dim(), 1)

class MTangent(sf.Matrix):
    SHAPE = (State.tangent_dim(), State.tangent_dim())
    
def vstate_to_state(v: VState):
    state = State.from_storage(v)
    # q_px4 = state["quat_nominal"].to_storage()
    # state["quat_nominal"] = sf.Rot3(sf.Quaternion(xyz=sf.V3(q_px4[1], q_px4[2], q_px4[3]), w=q_px4[0]))
    return state

def predict_covariance(
    state: VState,
    P: MTangent,
    accel: sf.V3,
    accel_var: sf.V3,
    gyro: sf.V3,
    gyro_var: sf.Scalar,
    dt: sf.Scalar
) -> MTangent:
    state = vstate_to_state(state)
    g = sf.Symbol("g") # does not appear in the jacobians

    # state_t = state.copy()
    
    noise = Values(
        accel = sf.V3.symbolic("a_n"),
        gyro = sf.V3.symbolic("w_n"),
    )

    # Nominal state kinematics
    input = Values(
        accel = accel - state["accel_bias"],
        gyro = gyro - state["gyro_bias"]
    )
    
    R = state["quat_nominal"]
    state_pred = state.copy()
    state_pred["quat_nominal"] = state["quat_nominal"] * sf.Rot3(sf.Quaternion(xyz=(input["gyro"] * dt / 2), w=1))
    state_pred["vel"] = state["vel"] + (R * input["accel"] + sf.V3(0, 0, g)) * dt
    state_pred["pos"] = state["pos"] + state["vel"] * dt
    
    # zero_state_error = {state[key]: state[key].zero() for key in state.keys()}
    # zero_noise = {noise[key]: noise[key].zero() for key in noise.keys()}
    A = VTangent(state_pred).jacobian(state)
    G = VTangent(state_pred).jacobian(input)

    var_u = sf.Matrix.diag([accel_var[0], accel_var[1], accel_var[2], gyro_var, gyro_var, gyro_var])
    P_new = A * P * A.T + G * var_u * G.T
    
    return P_new


# generate_px4_function(predict_covariance, output_names=None)


def predict_cov(
    p: sf.V3,
    v: sf.V3,
    R: sf.Rot3,
    b_a: sf.V3,
    b_g: sf.V3,
    P: MTangent,
    accel: sf.V3,
    accel_var: sf.V3,
    gyro: sf.V3,
    gyro_var: sf.V3,
    dt: sf.Scalar,
    g: sf.Scalar = sf.Symbol("g")
) -> MTangent:
    v_new = v + (R * (accel - b_a) + sf.V3(0, 0, g)) * dt
    p_new = p + v * dt
    R_new = R * sf.Rot3(sf.Quaternion(xyz=((gyro - b_g) * dt / 2), w=1))
    R_new_so3 = R * sf.Rot3.from_tangent((gyro - b_g) * dt)
    
    # print("R_new_so3", R_new_so3.jacobian(R))
    # print("*"*10)
    # print("R_new", R_new.jacobian(R))
    # 计算各雅可比矩阵
    jacobians = Values()
    
    # 1. 位置更新对状态的雅可比
    jacobians["dp_dp"] = p_new.jacobian(p)  # 3x3
    jacobians["dp_dv"] = p_new.jacobian(v)  # 3x3
    jacobians["dp_dR"] = p_new.jacobian(R)  # 3x9 (Rot3展开为9维)
    
    # 2. 速度更新对状态的雅可比
    jacobians["dv_dv"] = v_new.jacobian(v)  # 3x3
    jacobians["dv_dR"] = v_new.jacobian(R)  # 3x9
    jacobians["dv_db_a"] = v_new.jacobian(b_a)  # 3x3
    
    # 3. 旋转更新对状态的雅可比
    jacobians["dR_dR"] = R_new.jacobian(R)  # 9x9
    jacobians["dR_db_g"] = R_new.jacobian(b_g)  # 9x3
    
    # 打包新状态
    new_state = Values(
        p=p_new,
        v=v_new,
        R=R_new,
        b_a=b_a,
        b_g=b_g
    )
    # return new_state, jacobians

generate_px4_function(predict_cov, output_names=None)

