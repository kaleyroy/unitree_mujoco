import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading
import numpy as np

import config

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_ as LowCmd_default
from unitree_sdk2py.utils.crc import CRC

TOPIC_LOWCMD = "rt/lowcmd"

locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)

glfw = mujoco.glfw.glfw

linear_velocity = 0.0
angular_velocity = 0.0
max_linear_vel = 0.5
max_angular_vel = 0.5


def key_callback(keycode):
    global linear_velocity, angular_velocity
    
    if keycode == glfw.KEY_UP:
        linear_velocity = max_linear_vel
        angular_velocity = 0.0
    elif keycode == glfw.KEY_DOWN:
        linear_velocity = -max_linear_vel
        angular_velocity = 0.0
    elif keycode == glfw.KEY_LEFT:
        angular_velocity = max_angular_vel
    elif keycode == glfw.KEY_RIGHT:
        angular_velocity = -max_angular_vel
    elif keycode == glfw.KEY_SPACE:
        linear_velocity = 0.0
        angular_velocity = 0.0


class KeyboardGo2ControllerSDK:
    """
    依赖Unitree SDK的Go2机器人控制器（仿真版本）
    
    说明：
    - 发布命令：通过DDS发布LowCmd到rt/lowcmd话题
    - 获取状态：直接使用MuJoCo传感器数据（仿真环境）
    
    注意：在纯仿真环境中，没有其他节点发布lowstate数据，
    因此不能通过DDS订阅获取电机状态，必须使用MuJoCo传感器数据。
    """

    def __init__(self, mj_model, mj_data):
        self.mj_model = mj_model
        self.mj_data = mj_data

        self.num_motor = self.mj_model.nu

        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.max_linear_vel = 0.5
        self.max_angular_vel = 0.5

        self.standing_positions = np.array([
            0.0,  0.9, -1.8,
            0.0,  0.9, -1.8,
            0.0,  0.9, -1.8,
            0.0,  0.9, -1.8
        ])

        if self.num_motor > len(self.standing_positions):
            extended_positions = np.zeros(self.num_motor)
            extended_positions[:len(self.standing_positions)] = self.standing_positions
            self.standing_positions = extended_positions

        for i in range(min(self.num_motor, len(self.standing_positions))):
            self.mj_data.ctrl[i] = self.standing_positions[i]

        self.crc = CRC()
        self.low_cmd = LowCmd_default()
        self.init_low_cmd()

        ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
        self.lowcmd_publisher = ChannelPublisher(TOPIC_LOWCMD, LowCmd_)
        self.lowcmd_publisher.Init()

    def init_low_cmd(self):
        self.low_cmd.head[0] = 0xFE
        self.low_cmd.head[1] = 0xEF
        self.low_cmd.level_flag = 0xFF
        self.low_cmd.gpio = 0

        for i in range(self.num_motor):
            self.low_cmd.motor_cmd[i].mode = 0x01
            self.low_cmd.motor_cmd[i].q = 0.0
            self.low_cmd.motor_cmd[i].kp = 0.0
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kd = 0.0
            self.low_cmd.motor_cmd[i].tau = 0.0

    def _update_keyboard_input(self):
        global linear_velocity, angular_velocity
        
        self.linear_velocity = linear_velocity
        self.angular_velocity = angular_velocity

    def update(self):
        self._update_keyboard_input()
        self._apply_motor_control()
        self._send_lowcmd()

    def _apply_motor_control(self):
        if self.num_motor == 0:
            return

        current_positions = self.mj_data.sensordata[:self.num_motor].copy()
        current_velocities = self.mj_data.sensordata[self.num_motor:2*self.num_motor].copy()

        desired_positions = self.standing_positions[:self.num_motor].copy()

        if abs(self.linear_velocity) > 0.01 or abs(self.angular_velocity) > 0.01:
            current_time = time.time()
            walk_freq = 2.5

            phase_shifts = [0, np.pi, np.pi, 0]
            leg_indices = [(0,1,2), (3,4,5), (6,7,8), (9,10,11)]

            for i, (phase, indices) in enumerate(zip(phase_shifts, leg_indices)):
                leg_phase = (current_time * 2 * np.pi * walk_freq) % (2 * np.pi) + phase

                direction = np.sign(self.linear_velocity) if abs(self.linear_velocity) > 0.01 else 0

                swing_sign = np.sin(leg_phase)

                if swing_sign > 0:
                    hip_offset = 0.03 * swing_sign * direction
                    thigh_offset = 0.15 * swing_sign * direction
                    calf_offset = 0.3 * swing_sign * direction
                else:
                    hip_offset = 0.0
                    thigh_offset = 0.0
                    calf_offset = 0.0

                for j, idx in enumerate(indices):
                    if idx < len(desired_positions):
                        if j == 0:
                            desired_positions[idx] += hip_offset
                        elif j == 1:
                            desired_positions[idx] += thigh_offset
                        elif j == 2:
                            desired_positions[idx] += calf_offset

            if abs(self.angular_velocity) > 0.01:
                turn_sign = np.sign(self.angular_velocity)
                hip_turn_offset = 0.1 * turn_sign
                for i in range(4):
                    hip_idx = i * 3
                    if i < 2:
                        desired_positions[hip_idx] += hip_turn_offset
                    else:
                        desired_positions[hip_idx] -= hip_turn_offset

        kp = 80.0
        kd = 2.0

        for i in range(min(self.num_motor, len(desired_positions))):
            q_d = desired_positions[i]
            q = current_positions[i]
            dq = current_velocities[i]

            torque = kp * (q_d - q) + kd * (0.0 - dq)

            self.low_cmd.motor_cmd[i].q = q_d
            self.low_cmd.motor_cmd[i].kp = kp
            self.low_cmd.motor_cmd[i].dq = 0.0
            self.low_cmd.motor_cmd[i].kd = kd
            self.low_cmd.motor_cmd[i].tau = 0.0

            self.mj_data.ctrl[i] = torque

    def _send_lowcmd(self):
        self.low_cmd.crc = self.crc.Crc(self.low_cmd)
        self.lowcmd_publisher.Write(self.low_cmd)


def SimulationThread():
    global mj_data, mj_model

    controller = KeyboardGo2ControllerSDK(mj_model, mj_data)

    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        controller.update()

        mujoco.mj_step(mj_model, mj_data)

        locker.release()

        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    print("Keyboard Control Instructions (SDK Version with MuJoCo Keyboard):")
    print("- Use UP arrow key to move forward (continuous)")
    print("- Use DOWN arrow key to move backward (continuous)")
    print("- Use LEFT/RIGHT arrow keys to turn (continuous)")
    print("- Press SPACE to stop all movement")
    print("- Close the MuJoCo viewer window to exit")

    viewer = mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback)

    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()