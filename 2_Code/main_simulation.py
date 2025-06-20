"""太阳系三体问题模拟主程序：设置参数、控制模拟流程"""
import numpy as np
from numerical_methods import VerletIntegrator  # 导入核心算法
from data_analysis import calculate_orbital_elements, lyapunov_exponent
from visualization import plot_3d_orbit, plot_deviation, plot_mass_ratio_effect
import time

# ---------------------
# 1. 物理参数设置（参数化设计，便于修改）
# ---------------------
G = 6.67430e-11  # 万有引力常数 (N·m²/kg²)
# 天体质量 (kg)
m_sun = 1.989e30
m_earth = 5.972e24
m_moon = 7.342e22
# 初始位置 (m) - 日心坐标系
r_sun = np.array([0.0, 0.0, 0.0])
r_earth = np.array([1.496e11, 0.0, 0.0])  # 1 AU
r_moon = r_earth + np.array([3.844e8, 0.0, 0.0])  # 地月距离
# 初始速度 (m/s)
v_earth = np.array([0.0, 29783.0, 0.0])  # 地球公转速度
v_moon = v_earth + np.array([0.0, 1022.0, 0.0])  # 月球绕地速度
# 模拟参数
dt = 43200.0  # 时间步长 12小时 (s)
total_days = 10000  # 模拟总天数
output_interval = 100  # 输出数据间隔（天）

# ---------------------
# 2. 初始化积分器与天体系统
# ---------------------
def main():
    start_time = time.time()
    print("=== 太阳系三体问题模拟开始 ===")
    
    # 初始化Verlet积分器
    integrator = VerletIntegrator(G)
    # 添加天体：(质量, 位置, 速度)
    integrator.add_body(m_sun, r_sun, np.zeros(3))
    integrator.add_body(m_earth, r_earth, v_earth)
    integrator.add_body(m_moon, r_moon, v_moon)
    
    # ---------------------
    # 3. 执行模拟 - 标准组
    # ---------------------
    print("运行标准组模拟...")
    time_standard, positions_standard, velocities_standard = integrator.simulate(
        total_days=total_days, dt=dt, output_interval=output_interval
    )
    # 计算轨道要素
    a_earth_std, e_earth_std, a_moon_std, e_moon_std = calculate_orbital_elements(
        time_standard, positions_standard
    )
    # 保存标准组数据
    np.savez("3_Data/raw_data/standard_group.npz", 
             time=time_standard, 
             earth_pos=positions_standard[:, 1], 
             moon_pos=positions_standard[:, 2])
    
    # ---------------------
    # 4. 执行模拟 - 微扰组（月球速度+0.05%）
    # ---------------------
    print("运行微扰组模拟...")
    integrator.reset()  # 重置积分器
    integrator.add_body(m_sun, r_sun, np.zeros(3))
    integrator.add_body(m_earth, r_earth, v_earth)
    integrator.add_body(m_moon, r_moon, v_moon * 1.0005)  # 速度微扰
    time_perturb, positions_perturb, velocities_perturb = integrator.simulate(
        total_days=total_days, dt=dt, output_interval=output_interval
    )
    # 计算轨道要素
    a_earth_pert, e_earth_pert, a_moon_pert, e_moon_pert = calculate_orbital_elements(
        time_perturb, positions_perturb
    )
    # 保存微扰组数据
    np.savez("3_Data/raw_data/perturbed_group.npz", 
             time=time_perturb, 
             earth_pos=positions_perturb[:, 1], 
             moon_pos=positions_perturb[:, 2])
    
    # ---------------------
    # 5. 数据分析
    # ---------------------
    print("计算李雅普诺夫指数...")
    lyapunov = lyapunov_exponent(
        positions_standard[:, 2], positions_perturb[:, 2], 
        time_standard, dt=dt
    )
    print(f"最大李雅普诺夫指数: {lyapunov:.6f} 1/天")
    
    # ---------------------
    # 6. 可视化
    # ---------------------
    print("生成可视化图表...")
    # 3D轨道图
    plot_3d_orbit(
        time_standard, 
        positions_standard[:, 1], positions_standard[:, 2],
        positions_perturb[:, 2]
    )
    # 轨道偏差图
    plot_deviation(
        time_standard, 
        positions_standard[:, 2], positions_perturb[:, 2]
    )
    # 质量比效应图（需提前计算不同质量比数据）
    plot_mass_ratio_effect()
    
    end_time = time.time()
    print(f"模拟完成，总耗时: {end_time - start_time:.2f} 秒")

if __name__ == "__main__":
    main()
