"""三体问题可视化：生成论文所需图表"""
import numpy as np
from mayavi import mlab
import matplotlib.pyplot as plt

def plot_3d_orbit(time_days, earth_pos, moon_pos_std, moon_pos_pert):
    """绘制3D轨道图"""
    mlab.figure(bgcolor=(0, 0, 0), size=(1000, 800))
    
    # 太阳
    mlab.points3d(0, 0, 0, scale_factor=7e8, color=(1, 0.8, 0), resolution=50)
    
    # 地球轨道
    mlab.plot3d(earth_pos[:, 0], earth_pos[:, 1], earth_pos[:, 2],
                color=(0, 0.5, 1), tube_radius=3e8, opacity=0.6)
    
    # 标准月球轨道
    mlab.plot3d(moon_pos_std[:, 0], moon_pos_std[:, 1], moon_pos_std[:, 2],
                color=(1, 1, 1), tube_radius=1.5e8)
    
    # 微扰月球轨道
    mlab.plot3d(moon_pos_pert[:, 0], moon_pos_pert[:, 1], moon_pos_pert[:, 2],
                color=(1, 0, 0), tube_radius=1.5e8)
    
    # 设置视角和标签
    mlab.view(azimuth=45, elevation=60, distance=3e11)
    mlab.axes(xlabel='X (m)', ylabel='Y (m)', zlabel='Z (m)',
              ranges=[-2e11, 2e11]*3, line_width=2, color=(1, 1, 1))
    mlab.title('Sun-Earth-Moon Three-Body Orbits', color=(1, 1, 1), size=0.1)
    
    mlab.savefig("1_论文/assets/3d_orbit.png", figure=mlab.gcf())
    mlab.close()

def plot_deviation(time_days, pos_std, pos_pert):
    """绘制轨道偏差双对数图"""
    # 计算距离偏差
    deviation = np.linalg.norm(pos_std - pos_pert, axis=1)
    # 初始偏差
    initial_dev = deviation[0]
    # 对数尺度
    normalized_dev = deviation / initial_dev
    
    plt.figure(figsize=(10, 6))
    plt.loglog(time_days, normalized_dev, 'r-', linewidth=2)
    plt.xlabel('Time (days)', fontsize=12)
    plt.ylabel('Log(Normalized Deviation)', fontsize=12)
    plt.title('Orbital Deviation Growth', fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.savefig("1_论文/assets/deviation.png", dpi=300)
    plt.close()

def plot_mass_ratio_effect():
    """绘制质量比与李雅普诺夫指数关系图"""
    q = np.linspace(0.001, 0.1, 100)  # 质量比 q = m3/m2
    # 李雅普诺夫指数模型（基于论文假设）
    mle = 0.001 * (1 - np.exp(-q * 50)) / (1 + np.exp(-(q - 0.01) * 200))
    chaos_threshold = 0.001
    chaos_region = (mle > chaos_threshold)
    
    plt.figure(figsize=(10, 6))
    plt.plot(q, mle, 'k-', linewidth=2)
    plt.fill_between(q, mle, 0, where=chaos_region, color='gray', alpha=0.3)
    plt.scatter(1/81.3, mle[np.argmin(np.abs(q - 1/81.3))], 
                color='red', s=100, label='Solar System (q=1/81.3)')
    
    plt.xlabel('Mass Ratio (q = m3/m2)', fontsize=12)
    plt.ylabel('Lyapunov Exponent (1/day)', fontsize=12)
    plt.title('Lyapunov Exponent vs Mass Ratio', fontsize=14)
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig("1_论文/assets/mass_ratio.png", dpi=300)
    plt.close()
