import time
import mujoco
import numpy as np
import sys
import select
import termios
import tty
from threading import Thread
import threading

try:
    from pynput import keyboard as pynput_keyboard
    HAS_PYNPUT = True
    print(f"已导入 pynput 库")
except ImportError:
    HAS_PYNPUT = False
print(f"HAS_PYNPUT: {HAS_PYNPUT}")

try:
    import mujoco.viewer
    import glfw
    HEADLESS = False
except (ImportError, Exception):
    print("警告: 无法导入图形界面库，将运行在无头模式")
    HEADLESS = True
print(f"HEADLESS: {HEADLESS}")

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge

import config

# 全局变量用于键盘状态
key_states = {
    'up': False,
    'down': False,
    'left': False,
    'right': False,
}

def keyboard_callback(key):
    """全局键盘回调函数，处理方向键"""
    glfw = mujoco.glfw.glfw
    
    # MuJoCo 的回调只传递 key 值，不传递 action
    # 由于无法区分按下和释放，我们使用简单的按下逻辑
    # 注意：这种方法需要手动释放按键
    if key == glfw.KEY_UP:
        key_states['up'] = True
        print(f"[DEBUG] GLFW Key UP pressed, key_states: {key_states}")
    elif key == glfw.KEY_DOWN:
        key_states['down'] = True
        print(f"[DEBUG] GLFW Key DOWN pressed, key_states: {key_states}")
    elif key == glfw.KEY_LEFT:
        key_states['left'] = True
        print(f"[DEBUG] GLFW Key LEFT pressed, key_states: {key_states}")
    elif key == glfw.KEY_RIGHT:
        key_states['right'] = True
        print(f"[DEBUG] GLFW Key RIGHT pressed, key_states: {key_states}")
    # ESC键用于停止所有运动
    elif key == glfw.KEY_ESCAPE:
        key_states['up'] = False
        key_states['down'] = False
        key_states['left'] = False
        key_states['right'] = False
        print(f"[DEBUG] GLFW Key ESC pressed, stopping all movement")
    # 也可以使用空格键停止所有运动
    elif key == glfw.KEY_SPACE:
        key_states['up'] = False
        key_states['down'] = False
        key_states['left'] = False
        key_states['right'] = False
        print(f"[DEBUG] GLFW Key SPACE pressed, stopping all movement")

# 全局锁，用于保护共享资源
locker = threading.Lock()

# 加载MuJoCo模型和数据
mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)

# 调试：打印关节位置信息
print(f"mj_data.qpos shape: {mj_data.qpos.shape}")
print(f"mj_data.qpos initial: {mj_data.qpos}")
print(f"mj_model.nq: {mj_model.nq}")
print(f"mj_model.nu: {mj_model.nu}")

# 基于Unitree GO2标准站立姿势设置
# 参考Unitree官方文档，GO2站立姿势关节角度（弧度）
# 执行器顺序：FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, 
#             RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
initial_pose = np.array([
    0.0, -0.9, 1.8,     # 0-2: FR_hip, FR_thigh, FR_calf (前右腿)
    0.0, -0.9, 1.8,     # 3-5: FL_hip, FL_thigh, FL_calf (前左腿)
    0.0, -0.9, 1.8,     # 6-8: RR_hip, RR_thigh, RR_calf (后右腿)
    0.0, -0.9, 1.8,     # 9-11: RL_hip, RL_thigh, RL_calf (后左腿)
])

print(f"\nUnitree GO2标准站立姿势设置:")
print(f"关节角度: {initial_pose}")
print(f"髋关节: 0.0 rad (0°)")
print(f"大腿关节: -0.9 rad (-51.57°)")
print(f"小腿关节: 1.8 rad (103.13°)")

# 设置初始关节位置
if mj_model.nq >= 18:
    mj_data.qpos[6:18] = initial_pose
    print(f"已设置初始站立姿势到qpos[6:18]")

# 设置执行器控制
if mj_model.nu >= len(initial_pose):
    mj_data.ctrl[:len(initial_pose)] = initial_pose
    print(f"已设置执行器控制到initial_pose")

# 创建被动查看器，传入键盘回调函数
if not HEADLESS:
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=keyboard_callback)
else:
    viewer = None

# 设置模拟时间步长
mj_model.opt.timestep = config.SIMULATE_DT
num_motor_ = mj_model.nu
dim_motor_sensor_ = 3 * num_motor_

# 等待查看器初始化完成
if not HEADLESS:
    time.sleep(0.2)


class KeyboardController:
    """
    键盘控制器类，用于处理键盘输入并将其转换为机器人控制指令
    """
    
    def __init__(self):
        # 控制参数
        self.linear_velocity = 0.0  # 线速度 (前进/后退)
        self.angular_velocity = 0.0  # 角速度 (左转/右转)
        self.max_linear_vel = 0.5    # 最大线速度
        self.max_angular_vel = 0.5   # 最大角速度
        
        # 速度变化率
        self.linear_acceleration = 0.1  # 线加速度
        self.angular_acceleration = 0.1  # 角加速度
        
        # 行走相关参数
        self.walk_phase = 0.0  # 行走相位
        self.walk_frequency = 2.0  # 行走频率 (Hz)
        
        # Unitree GO2标准站立姿态关节角度 (弧度)
        # 执行器顺序: FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf, 
        #             RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
        self.stand_pose = np.array([
            0.0, -0.9, 1.8,     # 0-2: FR_hip, FR_thigh, FR_calf (前右腿)
            0.0, -0.9, 1.8,     # 3-5: FL_hip, FL_thigh, FL_calf (前左腿)
            0.0, -0.9, 1.8,     # 6-8: RR_hip, RR_thigh, RR_calf (后右腿)
            0.0, -0.9, 1.8,     # 9-11: RL_hip, RL_thigh, RL_calf (后左腿)
        ])

        
        # 电机控制增益 (基于Unitree标准参数)
        self.kp = 100.0  # 位置增益 (提高刚度以保持站立姿态)
        self.kd = 10.0   # 速度增益 (提高阻尼以稳定姿态)
        
        # 电机数量
        self.num_motor = 0
        
        # 按键状态（用于无头模式）
        self.key_states = {
            'up': False,
            'down': False,
            'left': False,
            'right': False,
        }
    
    def set_motor_num(self, num_motor):
        """设置电机数量"""
        self.num_motor = num_motor
        if len(self.stand_pose) != num_motor:
            # 扩展stand_pose到电机数量，只保留前12个关节的设置
            new_stand_pose = np.zeros(num_motor)
            new_stand_pose[:min(len(self.stand_pose), 12)] = self.stand_pose[:min(len(self.stand_pose), 12)]
            self.stand_pose = new_stand_pose
    
    def _get_key(self):
        """
        在无头模式下获取单个按键输入，不需要按回车
        """
        if select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], []):
            return sys.stdin.read(1)
        return None
    
    def _process_console_input(self):
        """
        处理控制台输入
        """
        key = self._get_key()
        if key:
            if key == 'w':
                self.key_states['up'] = True
                self.key_states['down'] = False
                print("前进")
            elif key == 's':
                self.key_states['down'] = True
                self.key_states['up'] = False
                print("后退")
            elif key == 'a':
                self.key_states['left'] = True
                self.key_states['right'] = False
                print("左转")
            elif key == 'd':
                self.key_states['right'] = True
                self.key_states['left'] = False
                print("右转")
            elif key == 'q':
                # 停止所有运动
                self.key_states['up'] = False
                self.key_states['down'] = False
                self.key_states['left'] = False
                self.key_states['right'] = False
                print("停止")
            elif key == 'x':
                # 退出程序
                print("退出程序")
                sys.exit(0)
    
    def _setup_keyboard_callback(self):
        """
        设置键盘回调函数，用于捕获键盘事件
        """
        print(f"[DEBUG] _setup_keyboard_callback called, HEADLESS={HEADLESS}, HAS_PYNPUT={HAS_PYNPUT}")
        
        if HEADLESS:
            print("无头模式：使用命令输入控制机器人")
            print("输入命令：w(前进) s(后退) a(左转) d(右转) q(停止) x(退出)")
            self.old_settings = termios.tcgetattr(sys.stdin)
            tty.setraw(sys.stdin.fileno())
            return
        
        if HAS_PYNPUT:
            self._setup_pynput_keyboard()
            print("已设置 pynput 键盘回调")
            return
        
        print("警告：无法设置键盘回调，将使用默认控制方式")
    
    def _setup_pynput_keyboard(self):
        """
        使用 pynput 库设置键盘监听
        """
        def on_press(key):
            try:
                if key == pynput_keyboard.Key.up:
                    self.key_states['up'] = True
                elif key == pynput_keyboard.Key.down:
                    self.key_states['down'] = True
                elif key == pynput_keyboard.Key.left:
                    self.key_states['left'] = True
                elif key == pynput_keyboard.Key.right:
                    self.key_states['right'] = True
                print(f"[DEBUG] Pynput Key pressed: {key}, key_states: {self.key_states}")
            except Exception as e:
                print(f"[DEBUG] Pynput Error in on_press: {e}")
        
        def on_release(key):
            try:
                if key == pynput_keyboard.Key.up:
                    self.key_states['up'] = False
                elif key == pynput_keyboard.Key.down:
                    self.key_states['down'] = False
                elif key == pynput_keyboard.Key.left:
                    self.key_states['left'] = False
                elif key == pynput_keyboard.Key.right:
                    self.key_states['right'] = False
                print(f"[DEBUG] Pynput Key released: {key}, key_states: {self.key_states}")
            except Exception as e:
                print(f"[DEBUG] Pynput Error in on_release: {e}")
        
        self.keyboard_listener = pynput_keyboard.Listener(on_press=on_press, on_release=on_release)
        self.keyboard_listener.start()
        print("[DEBUG] pynput keyboard listener started")
    
    def _setup_glfw_keyboard(self):
        """
        使用 GLFW 库设置键盘回调
        """
        glfw_window = glfw.CreateWindow(800, 600, "Keyboard Input", None, None)
        glfw.MakeContextCurrent(glfw_window)
        
        @glfw.SET_KEY_CALLBACK(glfw_window)
        def key_callback(window, key, scancode, action, mods):
            if key == glfw.KEY_UP:
                self.key_states['up'] = (action != glfw.RELEASE)
            elif key == glfw.KEY_DOWN:
                self.key_states['down'] = (action != glfw.RELEASE)
            elif key == glfw.KEY_LEFT:
                self.key_states['left'] = (action != glfw.RELEASE)
            elif key == glfw.KEY_RIGHT:
                self.key_states['right'] = (action != glfw.RELEASE)
            print(f"[DEBUG] Key pressed: {key}, action: {action}, key_states: {self.key_states}")
        
        print("[DEBUG] GLFW keyboard callback registered")
    
    def update_velocities(self):
        """
        根据当前按键状态更新速度值
        """
        # 同步全局按键状态到实例状态
        if not HEADLESS:
            # 在非无头模式下，同步全局按键状态
            # 这里我们只是读取当前的按键状态
            self.key_states['up'] = key_states['up']
            self.key_states['down'] = key_states['down']
            self.key_states['left'] = key_states['left']
            self.key_states['right'] = key_states['right']
        else:
            # 在无头模式下处理控制台输入
            self._process_console_input()
        
        current_key_states = self.key_states
        
        # 实现带衰减的速度控制，模拟按键释放效果
        # 每次更新时稍微降低速度，只有在按键时才加速
        self.linear_velocity *= 0.9  # 轻微的衰减
        self.angular_velocity *= 0.9  # 轻微的衰减
        
        # 根据当前按键状态调整速度
        if current_key_states['up'] and not current_key_states['down']:
            self.linear_velocity = self.max_linear_vel
        elif current_key_states['down'] and not current_key_states['up']:
            self.linear_velocity = -self.max_linear_vel
        else:
            # 如果没有相应的按键，逐渐减速到0
            if self.linear_velocity > 0.01:
                self.linear_velocity = max(0, self.linear_velocity - 0.05)
            elif self.linear_velocity < -0.01:
                self.linear_velocity = min(0, self.linear_velocity + 0.05)
            else:
                self.linear_velocity = 0.0  # 归零小数值
        
        # 更新角速度 (左转/右转) - 设定优先级：left > right > none
        if current_key_states['left'] and not current_key_states['right']:
            self.angular_velocity = self.max_angular_vel
        elif current_key_states['right'] and not current_key_states['left']:
            self.angular_velocity = -self.max_angular_vel
        else:
            # 如果没有相应的按键，逐渐减速到0
            if self.angular_velocity > 0.01:
                self.angular_velocity = max(0, self.angular_velocity - 0.05)
            elif self.angular_velocity < -0.01:
                self.angular_velocity = min(0, self.angular_velocity + 0.05)
            else:
                self.angular_velocity = 0.0  # 归零小数值
    
    def get_command(self):
        """
        获取当前控制指令，返回一个包含速度信息的字典
        
        Returns:
            dict: 包含线速度和角速度的字典
        """
        self.update_velocities()
        return {
            'lx': 0.0,  # 左摇杆X轴 (不使用)
            'ly': self.linear_velocity,  # 左摇杆Y轴 (前进/后退)
            'rx': 0.0,  # 右摇杆X轴 (不使用)
            'ry': self.angular_velocity,  # 右摇杆Y轴 (左转/右转)
        }
    
    def compute_motor_control(self, dt, current_q):
        """
        根据当前速度和姿态计算电机控制输入
        
        Args:
            dt: 时间步长
            current_q: 当前关节角度数组
            
        Returns:
            np.array: 电机控制力矩数组
        """
        self.update_velocities()
        
        if self.num_motor == 0:
            return np.zeros(12)
        
        # 计算是否有实际的运动命令（基于速度而非按键状态）
        has_movement = abs(self.linear_velocity) > 0.01 or abs(self.angular_velocity) > 0.01
        
        # 调试信息：打印按键状态和速度
        if has_movement:
            print(f"[DEBUG] 按键状态: {self.key_states}, 线速度: {self.linear_velocity:.3f}, 角速度: {self.angular_velocity:.3f}")
        
        # 计算目标关节角度 - 始终使用站立姿势作为基础
        target_q = self.stand_pose.copy()
        
        # 只有在有按键按下时才添加步态运动
        if has_movement and self.num_motor >= 12:
            # 更新行走相位
            speed = abs(self.linear_velocity) + abs(self.angular_velocity)
            if speed > 0.01:
                self.walk_phase += 2 * np.pi * self.walk_frequency * dt
                self.walk_phase = self.walk_phase % (2 * np.pi)
            else:
                self.walk_phase = 0.0
            
            # 添加迈步运动 (对角步态)
            if speed > 0.01:
                # FL 和 RR 腿相位相同，FR 和 RL 腿相位相同
                phase_offset = np.sin(self.walk_phase)
                
                # 髋关节摆动
                hip_swing = 0.15 * phase_offset * self.linear_velocity / self.max_linear_vel
                # 大腿摆动
                thigh_swing = 0.1 * phase_offset * self.linear_velocity / self.max_linear_vel
                # 小腿摆动
                calf_swing = -0.1 * phase_offset * self.linear_velocity / self.max_linear_vel
                
                # FR 腿 (前右) - 索引0-2
                target_q[0] += hip_swing
                target_q[1] += thigh_swing
                target_q[2] += calf_swing
                
                # FL 腿 (前左) - 索引3-5
                target_q[3] -= hip_swing
                target_q[4] -= thigh_swing
                target_q[5] -= calf_swing
                
                # RR 腿 (后右) - 索引6-8
                target_q[6] += hip_swing
                target_q[7] += thigh_swing
                target_q[8] += calf_swing
                
                # RL 腿 (后左) - 索引9-11
                target_q[9] -= hip_swing
                target_q[10] -= thigh_swing
                target_q[11] -= calf_swing
                
                # 偏航转向
                yaw_swing = 0.1 * self.angular_velocity / self.max_angular_vel
                target_q[0] += yaw_swing   # FR 髋关节
                target_q[3] += yaw_swing   # FL 髋关节
                target_q[6] -= yaw_swing   # RR 髋关节
                target_q[9] -= yaw_swing   # RL 髋关节
        else:
            # 没有按键时，重置行走相位
            self.walk_phase = 0.0
        
        # 获取关节速度用于PD控制
        if len(current_q) >= self.num_motor * 2:
            current_qvel = current_q[self.num_motor:self.num_motor*2]
        else:
            # 如果没有速度信息，使用估计值或零
            current_qvel = np.zeros(self.num_motor)
        
        # 计算控制力矩: torque = kp * (target_q - current_q) - kd * velocity
        pos_error = target_q - current_q[:self.num_motor]
        vel_feedback = self.kd * current_qvel
        
        ctrl = self.kp * pos_error - vel_feedback
        
        # 限制最大力矩
        max_torque = 20.0
        ctrl = np.clip(ctrl, -max_torque, max_torque)
        
        # 为了进一步提高稳定性，可以降低控制输出平滑性
        # 如果误差很小，可以适当减小控制力矩
        small_error_threshold = 0.1
        if np.all(np.abs(pos_error) < small_error_threshold):
            # 当接近目标时，稍微降低控制强度以减少震荡
            ctrl *= 0.7  # 减少30%的控制力矩以提高稳定性
        
        return ctrl
    
    def cleanup(self):
        """
        清理资源，恢复终端设置
        """
        if HEADLESS and self.old_settings:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)


class KeyboardUnitreeBridge(UnitreeSdk2Bridge):
    """
    继承自UnitreeSdk2Bridge的自定义桥接类，用于将键盘输入转换为机器人控制指令
    """
    
    def __init__(self, mj_model, mj_data):
        # 在调用父类初始化之前创建键盘控制器
        self.keyboard_controller = KeyboardController()
        
        # 调用父类初始化方法
        super().__init__(mj_model, mj_data)
        
        # 禁用手柄控制
        self.joystick = None
        
        # 设置电机数量
        self.keyboard_controller.set_motor_num(self.num_motor)
    
    def PublishWirelessController(self):
        """
        重写父类方法，发布键盘控制指令而不是手柄指令
        """
        # 获取键盘控制指令
        command = self.keyboard_controller.get_command()
        
        # 更新无线控制器消息
        # 注意：这里模拟了手柄的摇杆输入
        # print(f"lx: {command['lx']:.2f}, ly: {command['ly']:.2f}, rx: {command['rx']:.2f}, ry: {command['ry']:.2f}")
        self.wireless_controller.lx = command['lx']
        self.wireless_controller.ly = command['ly']
        self.wireless_controller.rx = command['rx']
        self.wireless_controller.ry = command['ry']
        
        # 设置按键状态为0，因为我们不使用按键控制
        self.wireless_controller.keys = 0
        
        # 发布控制指令
        self.wireless_controller_puber.Write(self.wireless_controller)
        
        # 直接设置电机控制输入
        self._apply_motor_control()
    
    def ApplyMotorControl(self):
        """
        应用电机控制，供主模拟线程调用
        """
        self._apply_motor_control()
    
    def _apply_motor_control(self):
        """
        直接应用电机控制到MuJoCo仿真
        """
        if self.mj_data is not None:
            # 获取当前关节状态 (位置和速度)
            # qpos包含所有广义坐标，包括关节位置和基座位置
            # qvel包含所有广义速度
            # 对于四足机器人，关节通常从索引6开始（前3个是位置，后3个是旋转）
            joint_pos = self.mj_data.qpos[6:6+self.num_motor]
            joint_vel = self.mj_data.qvel[6:6+self.num_motor]
            
            # 将位置和速度合并为一个数组传递给控制函数
            current_q = np.concatenate([joint_pos, joint_vel])
            
            # 调试输出
            lin_vel = self.keyboard_controller.linear_velocity
            ang_vel = self.keyboard_controller.angular_velocity
            key_states = self.keyboard_controller.key_states
            
            # 计算电机控制力矩
            ctrl = self.keyboard_controller.compute_motor_control(
                self.dt, current_q
            )
            
            # 调试: 打印按键状态和控制力矩
            if any(key_states.values()):
                print(f"Keys: {key_states}, lin_vel: {lin_vel:.2f}, ang_vel: {ang_vel:.2f}")
                print(f"  target_q: {self.keyboard_controller.stand_pose}")
                print(f"  ctrl: {ctrl}")
            
            # 设置电机控制输入
            # 确保控制向量的长度不超过实际的控制输入数量
            actual_ctrl_size = min(len(ctrl), len(self.mj_data.ctrl))
            self.mj_data.ctrl[:actual_ctrl_size] = ctrl[:actual_ctrl_size]
    
    def cleanup(self):
        """
        清理资源
        """
        if hasattr(self, 'keyboard_controller'):
            self.keyboard_controller.cleanup()


def SimulationThread():
    """
    模拟线程，负责运行物理模拟和更新机器人状态
    """
    global mj_data, mj_model

    # 初始化Unitree SDK通信
    # DOMAIN_ID用于区分不同的通信域，避免不同机器人之间的干扰
    # INTERFACE指定网络接口，"lo"表示本地回环接口
    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    
    # 创建自定义的桥接对象，用于连接MuJoCo模拟器和Unitree SDK
    unitree = KeyboardUnitreeBridge(mj_model, mj_data)

    # 打印场景信息（可选）
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    # 主模拟循环
    while viewer.is_running():
        # 记录步骤开始时间，用于控制模拟速度
        step_start = time.perf_counter()

        # 获取锁，保护共享资源
        locker.acquire()
        
        # 在物理模拟之前应用电机控制
        unitree.ApplyMotorControl()
        
        # 执行一步物理模拟
        # mj_step会根据当前状态和控制输入更新模拟器状态
        mujoco.mj_step(mj_model, mj_data)

        # 释放锁
        locker.release()

        # 计算到下一步的时间间隔，确保模拟速度与实际时间一致
        time_until_next_step = mj_model.opt.timestep - (
            time.perf_counter() - step_start
        )
        
        # 如果还有剩余时间，则等待
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)


def PhysicsViewerThread():
    """
    物理查看器线程，负责更新显示
    """
    if HEADLESS:
        # 在无头模式下，只打印状态信息
        print("无头模式：模拟器运行中...")
        try:
            while True:
                # 打印模拟状态
                print(f"模拟器正在运行... (按Ctrl+C退出)")
                time.sleep(5.0)
        except KeyboardInterrupt:
            print("\n接收到中断信号，正在退出...")
    else:
        while viewer.is_running():
            # 获取锁，保护共享资源
            locker.acquire()
            
            # 同步查看器，更新显示
            viewer.sync()
            
            # 释放锁
            locker.release()
            
            # 控制查看器更新频率
            time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    print("启动四足机器人键盘控制模拟器...")
    if not HEADLESS:
        print("使用方向键控制机器人:")
        print("  上键 - 前进")
        print("  下键 - 后退")
        print("  左键 - 左转")
        print("  右键 - 右转")
        print("按ESC键退出模拟器")
    
    # 创建并启动查看器线程
    viewer_thread = Thread(target=PhysicsViewerThread)
    
    # 创建并启动模拟线程
    sim_thread = Thread(target=SimulationThread)

    # 启动线程
    viewer_thread.start()
    sim_thread.start()
    
    try:
        # 等待线程结束
        viewer_thread.join()
        sim_thread.join()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    finally:
        # 清理资源
        if 'unitree' in locals():
            unitree.cleanup()
        print("程序已退出")