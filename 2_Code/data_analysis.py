import numpy as np

def calculate_total_energy(positions, velocities, masses):
    """计算系统总能量（动能+势能）"""
    # 动能
    kinetic_energy = 0.5 * np.sum([m * np.linalg.norm(v)**2 for m, v in zip(masses, velocities)])
    
    # 势能
    potential_energy = 0
    n_bodies = len(positions)
    
    for i in range(n_bodies):
        for j in range(i+1, n_bodies):
            r_ij = np.linalg.norm(positions[i] - positions[j])
            potential_energy -= 6.67430e-11 * masses[i] * masses[j] / r_ij
    
    return kinetic_energy + potential_energy

def calculate_orbit_elements(position, velocity, mass_primary):
    """计算轨道六要素（以主天体为中心）"""
    r = np.linalg.norm(position)
    v = np.linalg.norm(velocity)
    
    # 计算轨道能量和半长轴
    energy = 0.5 * v**2 - 6.67430e-11 * mass_primary / r
    a = -6.67430e-11 * mass_primary / (2 * energy) if energy < 0 else np.inf  # 椭圆轨道
    
    # 计算轨道角动量
    h = np.cross(position, velocity)
    h_norm = np.linalg.norm(h)
    
    # 计算偏心率向量
    e_vector = (1/(6.67430e-11 * mass_primary)) * ((v**2 - 6.67430e-11 * mass_primary / r) * position - np.dot(position, velocity) * velocity)
    e = np.linalg.norm(e_vector)
    
    # 计算倾角
    z_axis = np.array([0, 0, 1])
    i = np.arccos(np.dot(h, z_axis) / h_norm) if h_norm > 0 else 0
    
    # 计算升交点赤经
    n_vector = np.cross(z_axis, h)
    n_norm = np.linalg.norm(n_vector)
    Omega = np.arccos(n_vector[0] / n_norm) if n_norm > 0 else 0
    if n_vector[1] < 0:
        Omega = 2 * np.pi - Omega
    
    # 计算近点角距
    if e < 1 and n_norm > 0:  # 椭圆轨道
        w = np.arccos(np.dot(n_vector, e_vector) / (n_norm * e))
        if e_vector[2] < 0:
            w = 2 * np.pi - w
    else:
        w = 0
    
    return {
        'semi_major_axis': a,
        'eccentricity': e,
        'inclination': i,
        'ascending_node': Omega,
        'argument_of_periapsis': w
    }

def calculate_lyapunov_exponent(trajectory1, trajectory2, time_step):
    """计算李雅普诺夫指数"""
    distances = [np.linalg.norm(traj1[1] - traj2[1]) for traj1, traj2 in zip(trajectory1, trajectory2)]
    initial_distance = distances[0]
    
    # 对数距离
    log_distances = [np.log(dist / initial_distance) for dist in distances]
    times = [i * time_step / 86400 for i in range(len(distances))]  # 转换为天
    
    # 线性拟合（前半段数据）
    valid_indices = int(len(times) * 0.5)
    slope, _ = np.polyfit(times[:valid_indices], log_distances[:valid_indices], 1)
    
    return slope, times, log_distances
