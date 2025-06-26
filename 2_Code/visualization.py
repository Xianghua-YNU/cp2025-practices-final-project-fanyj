import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import os

def plot_orbits(trajectories, title="三体系统轨道", save_path=None):
    """绘制3D轨道图"""
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取轨迹数据
    sun_traj = np.array([traj[0] for traj in trajectories]) / 1.496e11  # 转换为AU
    earth_traj = np.array([traj[1] for traj in trajectories]) / 1.496e11
    moon_traj = np.array([traj[2] for traj in trajectories]) / 1.496e11
    
    # 绘制轨道
    ax.plot(sun_traj[:, 0], sun_traj[:, 1], sun_traj[:, 2], 'yo-', label='太阳')
    ax.plot(earth_traj[:, 0], earth_traj[:, 1], earth_traj[:, 2], 'b-', label='地球')
    ax.plot(moon_traj[:, 0], moon_traj[:, 1], moon_traj[:, 2], 'g-', label='月球')
    
    # 设置坐标轴
    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    ax.set_title(title)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_energy_conservation(energies, initial_energy, title="能量守恒", save_path=None):
    """绘制能量守恒图"""
    time = np.arange(len(energies)) * 12 * 3600 / 86400  # 转换为天
    energy_ratio = np.array(energies) / initial_energy
    
    plt.figure(figsize=(10, 6))
    plt.plot(time, energy_ratio)
    plt.title(title)
    plt.xlabel('时间 (天)')
    plt.ylabel('E/E0')
    plt.grid(True)
    plt.ylim(0.999, 1.001)  # 放大显示能量变化
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_orbit_elements(orbit_data, title="轨道要素演化", save_path=None):
    """绘制轨道要素演化图"""
    time = np.array([data['time'] for data in orbit_data])
    
    # 地球半长轴 (AU)
    earth_a = np.array([data['earth']['semi_major_axis'] / 1.496e11 for data in orbit_data])
    # 地球偏心率
    earth_e = np.array([data['earth']['eccentricity'] for data in orbit_data])
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))
    fig.suptitle(title, fontsize=14)
    
    axes[0].plot(time, earth_a)
    axes[0].set_title('地球半长轴 (AU)')
    axes[0].set_xlabel('时间 (天)')
    axes[0].grid(True)
    
    axes[1].plot(time, earth_e)
    axes[1].set_title('地球偏心率')
    axes[1].set_xlabel('时间 (天)')
    axes[1].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def generate_animation(trajectories, title="三体系统轨道演化", filename="animation.gif", show_progress=True):
    """生成三体运动动画"""
    import time
    start_time = time.time()
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 提取轨迹数据
    skip = max(1, len(trajectories) // 200)  # 最多200帧
    sun_traj = np.array([traj[0] for traj in trajectories[::skip]]) / 1.496e11
    earth_traj = np.array([traj[1] for traj in trajectories[::skip]]) / 1.496e11
    moon_traj = np.array([traj[2] for traj in trajectories[::skip]]) / 1.496e11
    time_steps = len(sun_traj)
    
    # 绘制初始帧
    sun, = ax.plot([sun_traj[0, 0]], [sun_traj[0, 1]], [sun_traj[0, 2]], 'yo', markersize=10, label='太阳')
    earth, = ax.plot([earth_traj[0, 0]], [earth_traj[0, 1]], [earth_traj[0, 2]], 'bo', markersize=6, label='地球')
    moon, = ax.plot([moon_traj[0, 0]], [moon_traj[0, 1]], [moon_traj[0, 2]], 'go', markersize=4, label='月球')
    earth_orbit, = ax.plot(earth_traj[:, 0], earth_traj[:, 1], earth_traj[:, 2], 'b-', alpha=0.3)
    moon_orbit, = ax.plot(moon_traj[:, 0], moon_traj[:, 1], moon_traj[:, 2], 'g-', alpha=0.3)
    
    # 设置坐标轴
    ax.set_xlabel('X (AU)')
    ax.set_ylabel('Y (AU)')
    ax.set_zlabel('Z (AU)')
    ax.set_title(title)
    ax.legend()
    
    # 自动调整视图范围
    max_range = np.max([
        np.max(earth_traj[:, 0]) - np.min(earth_traj[:, 0]),
        np.max(earth_traj[:, 1]) - np.min(earth_traj[:, 1]),
        np.max(earth_traj[:, 2]) - np.min(earth_traj[:, 2])
    ]) * 1.1
    mid_x = 0.5 * (np.max(earth_traj[:, 0]) + np.min(earth_traj[:, 0]))
    mid_y = 0.5 * (np.max(earth_traj[:, 1]) + np.min(earth_traj[:, 1]))
    mid_z = 0.5 * (np.max(earth_traj[:, 2]) + np.min(earth_traj[:, 2]))
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    # 动画更新函数
    def update(frame):
        if show_progress and frame % 10 == 0:
            print(f"生成动画帧 {frame}/{time_steps}")
        
        sun.set_data([sun_traj[frame, 0]], [sun_traj[frame, 1]])
        sun.set_3d_properties([sun_traj[frame, 2]])
        earth.set_data([earth_traj[frame, 0]], [earth_traj[frame, 1]])
        earth.set_3d_properties([earth_traj[frame, 2]])
        moon.set_data([moon_traj[frame, 0]], [moon_traj[frame, 1]])
        moon.set_3d_properties([moon_traj[frame, 2]])
        earth_orbit.set_data(earth_traj[:frame+1, 0], earth_traj[:frame+1, 1])
        earth_orbit.set_3d_properties(earth_traj[:frame+1, 2])
        moon_orbit.set_data(moon_traj[:frame+1, 0], moon_traj[:frame+1, 1])
        moon_orbit.set_3d_properties(moon_traj[:frame+1, 2])
        ax.set_title(f"{title} (时间: {frame*skip*12/24:.1f}天)")
        return sun, earth, moon, earth_orbit, moon_orbit
    
    # 生成动画
    anim = FuncAnimation(fig, update, frames=time_steps, interval=50, blit=True)
    
    # 保存动画
    if not filename.lower().endswith('.gif'):
        filename = os.path.splitext(filename)[0] + '.gif'
    
    try:
        anim.save(filename, writer='pillow', fps=20, dpi=100)
        if show_progress:
            print(f"动画已保存至 {filename}")
    except Exception as e:
        print(f"保存动画失败: {e}")
        print("请确保已安装Pillow库 (pip install pillow)")
    
    plt.close()
    return filename
