import time
import mujoco
import mujoco.viewer
from threading import Thread
import threading
import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge

import config


locker = threading.Lock()

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)

# Global variable to track keyboard states
key_states = {
    'up': False,
    'down': False,
    'left': False,
    'right': False,
}

def keyboard_callback(key):
    """Keyboard callback function to handle arrow keys"""
    try:
        glfw = mujoco.glfw.glfw
        
        if key == glfw.KEY_UP:
            key_states['up'] = True
        elif key == glfw.KEY_DOWN:
            key_states['down'] = True
        elif key == glfw.KEY_LEFT:
            key_states['left'] = True
        elif key == glfw.KEY_RIGHT:
            key_states['right'] = True
        elif key == glfw.KEY_ESCAPE or key == glfw.KEY_SPACE:
            # Stop all movement
            key_states['up'] = False
            key_states['down'] = False
            key_states['left'] = False
            key_states['right'] = False
    except AttributeError:
        # Handle case where glfw is not available
        pass

# Check if we should enable elastic band (similar to original)
if config.ENABLE_ELASTIC_BAND:
    from unitree_sdk2py_bridge import ElasticBand
    elastic_band = ElasticBand()
    if config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
else:
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=keyboard_callback
    )

mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

time.sleep(0.2)


class KeyboardUnitreeBridge(UnitreeSdk2Bridge):
    """
    Custom bridge class that handles keyboard input instead of joystick input
    """
    
    def __init__(self, mj_model, mj_data):
        # Call parent constructor first to ensure proper initialization
        super().__init__(mj_model, mj_data)
        
        self.linear_velocity = 0.0  # Forward/backward movement
        self.angular_velocity = 0.0  # Turning movement
        self.max_linear_vel = 0.5    # Maximum linear velocity
        self.max_angular_vel = 0.5   # Maximum angular velocity
        
        # Define standard standing pose for Go2 robot
        # These are the desired joint positions when robot is standing
        self.standing_positions = np.array([
            0.0,  -0.3,  0.6,  # FR (Front Right): hip, thigh, calf
            0.0,  -0.3,  0.6,  # FL (Front Left): hip, thigh, calf  
            0.0,  -0.3,  0.6,  # RR (Rear Right): hip, thigh, calf
            0.0,  -0.3,  0.6   # RL (Rear Left): hip, thigh, calf
        ])
        
        # Extend to match the number of motors if needed
        if self.num_motor > len(self.standing_positions):
            extended_positions = np.zeros(self.num_motor)
            extended_positions[:len(self.standing_positions)] = self.standing_positions
            self.standing_positions = extended_positions
        
        # Set initial control values to standing pose
        for i in range(min(self.num_motor, len(self.standing_positions))):
            self.mj_data.ctrl[i] = self.standing_positions[i]
    
    def PublishWirelessController(self):
        """
        Override parent method to publish keyboard-based control commands
        instead of joystick commands
        """
        # Update velocities based on key states
        self._update_velocities()
        
        # Map keyboard inputs to wireless controller values
        # Up/Down arrows control forward/backward movement (ly axis)
        # Left/Right arrows control turning (rx axis)
        self.wireless_controller.lx = 0.0  # No sideways movement
        self.wireless_controller.ly = self.linear_velocity
        self.wireless_controller.rx = self.angular_velocity
        self.wireless_controller.ry = 0.0  # No pitch movement
        
        # Reset keys to 0 since we're not using button controls
        self.wireless_controller.keys = 0
        
        # Publish the control command
        self.wireless_controller_puber.Write(self.wireless_controller)
        
        # Apply motor control based on current state and commands
        self._apply_motor_control()
    
    def _update_velocities(self):
        """
        Update velocity values based on current key states
        """
        global key_states
        
        # Simple mapping: up/down for forward/backward, left/right for turning
        if key_states['up'] and not key_states['down']:
            self.linear_velocity = self.max_linear_vel
        elif key_states['down'] and not key_states['up']:
            self.linear_velocity = -self.max_linear_vel
        else:
            self.linear_velocity = 0.0  # Stop when no directional keys are pressed
            
        if key_states['left'] and not key_states['right']:
            self.angular_velocity = self.max_angular_vel
        elif key_states['right'] and not key_states['left']:
            self.angular_velocity = -self.max_angular_vel
        else:
            self.angular_velocity = 0.0  # Stop turning when no directional keys are pressed
    
    def _apply_motor_control(self):
        """
        Apply motor control based on current velocities and desired pose
        """
        if self.num_motor == 0:
            return

        # Use PD control to achieve desired positions
        current_positions = self.mj_data.sensordata[:self.num_motor] if len(self.mj_data.sensordata) >= self.num_motor else np.zeros(self.num_motor)
        
        # Base desired positions (standing pose)
        desired_positions = self.standing_positions[:self.num_motor].copy()
        
        # Add movement commands to desired positions if moving
        if abs(self.linear_velocity) > 0.01 or abs(self.angular_velocity) > 0.01:
            # Generate walking gait pattern when moving
            current_time = time.time()
            walk_freq = 2.0  # Walking frequency in Hz
            
            # Define phase shifts for each leg (to create walking motion)
            phase_shifts = [0, np.pi, np.pi/2, 3*np.pi/2]  # FR, FL, RR, RL
            leg_indices = [(0,1,2), (3,4,5), (6,7,8), (9,10,11)]
            
            for i, (phase, indices) in enumerate(zip(phase_shifts, leg_indices)):
                if i < len(leg_indices):  # Make sure we don't go out of bounds
                    leg_phase = (current_time * 2 * np.pi * walk_freq) % (2 * np.pi) + phase
                    
                    # Add walking motion to leg joints
                    if len(indices) <= len(desired_positions):
                        # Hip oscillation for walking
                        hip_offset = 0.1 * np.sin(leg_phase) * np.sign(self.linear_velocity)
                        
                        # Thigh and calf adjustments for stepping motion
                        thigh_offset = 0.2 * np.sin(leg_phase + np.pi/2) * np.sign(self.linear_velocity)
                        calf_offset = -0.2 * np.sin(leg_phase + np.pi/2) * np.sign(self.linear_velocity)
                        
                        for j, idx in enumerate(indices):
                            if idx < len(desired_positions):
                                if j == 0:  # Hip
                                    desired_positions[idx] += hip_offset
                                elif j == 1:  # Thigh
                                    desired_positions[idx] += thigh_offset
                                elif j == 2:  # Calf
                                    desired_positions[idx] += calf_offset
        
        # Simple PD controller
        kp = 50.0  # Position gain
        kd = 2.0   # Velocity gain (damping)
        
        # Calculate position error
        pos_error = desired_positions - current_positions
        
        # Get current velocities if available
        if len(self.mj_data.sensordata) >= self.num_motor * 2:
            current_velocities = self.mj_data.sensordata[self.num_motor:self.num_motor*2]
        else:
            current_velocities = np.zeros(self.num_motor)
        
        # Calculate control effort: tau = kp*pos_error - kd*vel_feedback
        control_effort = kp * pos_error - kd * current_velocities
        
        # Apply limits to control effort
        max_tau = 5.0  # Maximum torque
        control_effort = np.clip(control_effort, -max_tau, max_tau)
        
        # Apply control to the motors
        for i in range(min(self.num_motor, len(control_effort))):
            self.mj_data.ctrl[i] = control_effort[i]


def SimulationThread():
    global mj_data, mj_model

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = KeyboardUnitreeBridge(mj_model, mj_data)

    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    start_time = time.time()
    
    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()

        # Update simulation time for startup stabilization
        current_time = time.time() - start_time
        unitree.simulation_time = current_time
        
        # Publish wireless controller commands based on keyboard input and apply motor control
        unitree.PublishWirelessController()
        
        # Apply elastic band if enabled (similar to original)
        if config.ENABLE_ELASTIC_BAND:
            if 'elastic_band' in globals() and elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )
        
        # Step the simulation
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
    print("- Use UP/DOWN arrow keys to move forward/backward")
    print("- Use LEFT/RIGHT arrow keys to turn left/right")
    print("- Press SPACE or ESC to stop movement")
    print("- Close the window to exit")
    
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()