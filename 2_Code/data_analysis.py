"""三体问题数据分析：计算轨道要素、李雅普诺夫指数等"""
import numpy as np
from scipy.optimize import curve_fit

def calculate_orbital_elements(time_days, positions):
    """计算地球和月球的轨道要素（简化版：仅算半长轴和偏心率）"""
    # positions形状: (时间步, 天体索引, xyz)
    # 假设索引1是地球，索引2是月球（根据main.py的设置）
    earth_pos = positions[:, 1]
    moon_pos = positions[:, 2]
    
    # 计算地球轨道半长轴和偏心率（简化：假设轨道为椭圆）
    r_earth = np.linalg.norm(earth_pos, axis=1)
    a_earth = np.mean(r_earth)  # 简化：用平均距离作为半长轴
    e_earth = np.std(r_earth) / a_earth  # 简化：用标准差估计偏心率
    
    # 计算月球轨道半长轴和偏心率
    r_moon = np.linalg.norm(moon_pos - earth_pos, axis=1)
    a_moon = np.mean(r_moon)
    e_moon = np.std(r_moon) / a_moon
    
    return a_earth, e_earth, a_moon, e_moon

def lyapunov_exponent(pos1, pos2, time_days, dt):
    """计算最大李雅普诺夫指数（Wolf算法简化版）"""
    # pos1, pos2: 两组轨道位置 (时间步, xyz)
    # 计算初始距离
    delta_r0 = np.linalg.norm(pos1[0] - pos2[0])
    if delta_r0 < 1e-10:
        raise ValueError("初始距离过小，无法计算李雅普诺夫指数")
    
    # 选择等间隔的时间点（至少10个点）
    num_points = min(10, len(time_days))
    indices = np.linspace(0, len(time_days)-1, num_points, dtype=int)
    
    # 计算各时间点的距离
    delta_r = []
    for idx in indices:
        dr = np.linalg.norm(pos1[idx] - pos2[idx])
        delta_r.append(dr)
    
    # 拟合 ln(delta_r/delta_r0) = lambda * t
    def fit_func(t, lam):
        return lam * t
    
    # 转换为对数尺度
    t_fit = time_days[indices]
    y_fit = np.log(np.array(delta_r) / delta_r0)
    
    # 线性拟合
    popt, _ = curve_fit(fit_func, t_fit, y_fit)
    return popt[0]  # 返回李雅普诺夫指数
