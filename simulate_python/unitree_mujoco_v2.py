import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading
import numpy as np
import pygame

import config


locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)

viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


class KeyboardGo2Controller:
    """
    完全自主控制Go2机器人的控制器，不依赖Unitree SDK
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

        pygame.init()
        pygame.display.set_mode((200, 100))
        pygame.display.set_caption("Keyboard Control")

    def _update_keyboard_input(self):
        for event in pygame.event.get():
            print(f"event: {event}")
            pass

        keys = pygame.key.get_pressed()

        key_up = keys[pygame.K_UP]
        key_down = keys[pygame.K_DOWN]
        key_left = keys[pygame.K_LEFT]
        key_right = keys[pygame.K_RIGHT]

        if key_up and not key_down:
            self.linear_velocity = self.max_linear_vel
        elif key_down and not key_up:
            self.linear_velocity = -self.max_linear_vel
        else:
            self.linear_velocity = 0.0

        if key_left and not key_right:
            self.angular_velocity = self.max_angular_vel
        elif key_right and not key_left:
            self.angular_velocity = -self.max_angular_vel
        else:
            self.angular_velocity = 0.0

    def update(self):
        self._update_keyboard_input()
        self._apply_motor_control()

    def _apply_motor_control(self):
        if self.num_motor == 0:
            return

        current_positions = self.mj_data.sensordata[:self.num_motor].copy() if len(self.mj_data.sensordata) >= self.num_motor else np.zeros(self.num_motor)
        desired_positions = self.standing_positions[:self.num_motor].copy()

        if abs(self.linear_velocity) > 0.01 or abs(self.angular_velocity) > 0.01:
            current_time = time.time()
            walk_freq = 2.5

            phase_shifts = [0, np.pi, np.pi, 0]
            leg_indices = [(0,1,2), (3,4,5), (6,7,8), (9,10,11)]

            for i, (phase, indices) in enumerate(zip(phase_shifts, leg_indices)):
                leg_phase = (current_time * 2 * np.pi * walk_freq) % (2 * np.pi) + phase

                direction = np.sign(self.linear_velocity) if abs(self.linear_velocity) > 0.01 else 0

                swing_height = 0.08
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

            if len(self.mj_data.sensordata) >= self.num_motor * 2:
                dq = self.mj_data.sensordata[i + self.num_motor]
            else:
                dq = 0.0

            self.mj_data.ctrl[i] = kp * (q_d - q) + kd * (0.0 - dq)


def SimulationThread():
    global mj_data, mj_model

    controller = KeyboardGo2Controller(mj_model, mj_data)

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
    print("Keyboard Control Instructions:")
    print("- Make sure the 'Keyboard Control' window is active (focused)")
    print("- Use UP arrow key to move LEFT (X-axis negative)")
    print("- Use DOWN arrow key to move RIGHT (X-axis positive)")
    print("- Use LEFT/RIGHT arrow keys to turn")
    print("- Close the windows to exit")

    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()