#!/usr/bin/env python3
"""
最简化的测试程序，用于验证基本功能
"""

import time
import mujoco
import numpy as np
import sys

# 尝试导入必要的模块
try:
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from unitree_sdk2py_bridge import UnitreeSdk2Bridge
    import config
    print("成功导入所有必要模块")
except ImportError as e:
    print(f"导入模块失败: {e}")
    sys.exit(1)

# 加载MuJoCo模型和数据
try:
    mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
    mj_data = mujoco.MjData(mj_model)
    print(f"成功加载机器人模型，电机数量: {mj_model.nu}")
except Exception as e:
    print(f"加载模型失败: {e}")
    sys.exit(1)

# 设置模拟时间步长
mj_model.opt.timestep = config.SIMULATE_DT
print(f"设置模拟时间步长: {config.SIMULATE_DT}")

# 初始化Unitree SDK通信
try:
    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    print("成功初始化Unitree SDK通信")
except Exception as e:
    print(f"初始化Unitree SDK失败: {e}")
    sys.exit(1)

# 创建桥接对象
try:
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)
    print("成功创建Unitree桥接对象")
except Exception as e:
    print(f"创建Unitree桥接对象失败: {e}")
    sys.exit(1)

# 运行几步模拟
print("开始模拟...")
for i in range(5):
    step_start = time.perf_counter()
    mujoco.mj_step(mj_model, mj_data)
    elapsed = time.perf_counter() - step_start
    print(f"步骤 {i+1} 完成，耗时: {elapsed:.6f}秒")
    
    # 计算到下一步的时间间隔
    time_until_next_step = mj_model.opt.timestep - elapsed
    if time_until_next_step > 0:
        time.sleep(time_until_next_step)

print("模拟测试完成")