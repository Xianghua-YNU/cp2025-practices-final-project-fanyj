import numpy as np

# 物理常数定义
G = 6.67430e-11  # 万有引力常数 (N·m²/kg²)
MSUN = 1.989e30  # 太阳质量 (kg)
EARTH_MASS = 5.972e24  # 地球质量 (kg)
MOON_MASS = 7.348e22  # 月球质量 (kg)
AU = 1.496e11  # 天文单位 (m)
DAY = 86400  # 1天的秒数 (s)

class VerletIntegrator:
    """通用速度Verlet积分器，适用于任何N体系统"""
    
    def __init__(self, masses, initial_positions, initial_velocities, 
                 time_step, force_function):
        """
        初始化Verlet积分器
        masses: 天体质量数组 [m1, m2, ...] (kg)
        initial_positions: 初始位置数组 [r1, r2, ...] (m)
        initial_velocities: 初始速度数组 [v1, v2, ...] (m/s)
        time_step: 时间步长 (s)
        force_function: 计算力的函数，应接受位置数组并返回加速度数组
        """
        self.masses = np.array(masses, dtype=float)
        self.positions = np.array(initial_positions, dtype=float)
        self.velocities = np.array(initial_velocities, dtype=float)
        self.time_step = time_step
        self.force_function = force_function
        self.accelerations = self.force_function(self.positions, self.masses)
        self.trajectories = [self.positions.copy()]
        self.time = 0
    
    def step(self):
        """执行一个时间步的Verlet算法"""
        # 位置预测
        new_positions = self.positions + self.velocities * self.time_step + 0.5 * self.accelerations * self.time_step**2
        
        # 计算新加速度
        self.positions = new_positions
        new_accelerations = self.force_function(self.positions, self.masses)
        
        # 速度校正
        self.velocities += 0.5 * (self.accelerations + new_accelerations) * self.time_step
        self.accelerations = new_accelerations
        
        # 保存轨迹
        self.trajectories.append(self.positions.copy())
        self.time += self.time_step

def calculate_gravitational_acceleration(positions, masses):
    """计算万有引力产生的加速度"""
    n_bodies = len(positions)
    acc = np.zeros_like(positions)
    
    for i in range(n_bodies):
        for j in range(n_bodies):
            if i != j:
                r_ij = positions[j] - positions[i]
                r_ij_norm = np.linalg.norm(r_ij)
                acc[i] += G * masses[j] * r_ij / (r_ij_norm ** 3)
    
    return acc

def setup_solar_system_initial_conditions(perturbation=0.0):
    """
    设置太阳系三体问题初始条件，可添加速度扰动
    perturbation: 月球速度扰动百分比 (0.05表示+0.05%)
    """
    # 质量数组
    masses = [MSUN, EARTH_MASS, MOON_MASS]
    
    # 初始位置 (m)
    earth_pos = np.array([AU, 0, 0])
    moon_rel_pos = np.array([384400000, 0, 0])
    sun_pos = np.array([0, 0, 0])
    
    initial_positions = [sun_pos, earth_pos, earth_pos + moon_rel_pos]
    
    # 初始速度 (m/s)
    earth_vel = np.array([0, 29783, 0])
    moon_rel_vel = np.array([0, 1022, 0])
    moon_vel = moon_rel_vel * (1 + perturbation/100) + earth_vel
    sun_vel = np.array([0, 0, 0])
    
    initial_velocities = [sun_vel, earth_vel, moon_vel]
    
    return masses, initial_positions, initial_velocities
