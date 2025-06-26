from numerical_methods import VerletIntegrator, setup_solar_system_initial_conditions, calculate_gravitational_acceleration
from data_analysis import calculate_total_energy, calculate_orbit_elements, calculate_lyapunov_exponent
from visualization import plot_orbits, plot_energy_conservation, plot_orbit_elements, generate_animation
from utils import create_results_directory, timer_decorator
import numpy as np

@timer_decorator
def run_simulation(perturbation=0.0, days=1000, time_step=12*3600):
    """运行三体系统模拟"""
    print(f"运行模拟 (扰动: {perturbation}%, 时长: {days}天)...")
    
    # 设置初始条件
    masses, positions, velocities = setup_solar_system_initial_conditions(perturbation)
    
    # 初始化积分器
    integrator = VerletIntegrator(
        masses=masses,
        initial_positions=positions,
        initial_velocities=velocities,
        time_step=time_step,
        force_function=calculate_gravitational_acceleration
    )
    
    # 运行模拟
    steps = int(days * 86400 / time_step)
    energies = [calculate_total_energy(positions, velocities, masses)]
    orbit_elements = []
    
    for i in range(steps):
        if i % 1000 == 0:
            print(f"模拟进度: {i/steps*100:.1f}%")
            
            # 计算当前轨道要素
            earth_pos = positions[1] - positions[0]  # 相对太阳的位置
            earth_vel = velocities[1]
            earth_orbit = calculate_orbit_elements(earth_pos, earth_vel, masses[0])
            
            moon_pos = positions[2] - positions[0]
            moon_vel = velocities[2]
            moon_orbit = calculate_orbit_elements(moon_pos, moon_vel, masses[0])
            
            orbit_elements.append({
                'time': i * time_step / 86400,  # 转换为天
                'earth': earth_orbit,
                'moon': moon_orbit
            })
        
        # 执行一步积分
        integrator.step()
        
        # 计算并记录能量
        current_energy = calculate_total_energy(
            integrator.positions, 
            integrator.velocities, 
            integrator.masses
        )
        energies.append(current_energy)
    
    print("模拟完成!")
    
    return {
        'trajectories': integrator.trajectories,
        'energies': energies,
        'orbit_elements': orbit_elements,
        'masses': masses
    }

def compare_simulations(standard_results, perturbed_results, time_step):
    """比较标准模拟和扰动模拟的结果"""
    print("分析李雅普诺夫指数...")
    
    # 计算李雅普诺夫指数
    lyapunov, times, log_distances = calculate_lyapunov_exponent(
        standard_results['trajectories'],
        perturbed_results['trajectories'],
        time_step
    )
    
    print(f"李雅普诺夫指数: {lyapunov:.8f} /天")
    return lyapunov, times, log_distances

def main():
    """主函数：运行模拟并生成结果"""
    # 创建结果目录
    create_results_directory()
    
    # 设置模拟参数
    time_step = 12 * 3600  # 12小时
    simulation_days = 1000  # 模拟1000天
    
    # 运行标准模拟（无扰动）
    standard_results = run_simulation(
        perturbation=0.0,
        days=simulation_days,
        time_step=time_step
    )
    
    # 运行扰动模拟（0.05%速度扰动）
    perturbed_results = run_simulation(
        perturbation=0.05,
        days=simulation_days,
        time_step=time_step
    )
    
    # 比较两个模拟结果
    lyapunov, times, log_distances = compare_simulations(
        standard_results, 
        perturbed_results,
        time_step
    )
    
    # 生成可视化结果
    print("生成可视化结果...")
    
    # 绘制轨道图
    plot_orbits(
        standard_results['trajectories'], 
        title="标准模拟轨道", 
        save_path="standard_orbits.png"
    )
    
    plot_orbits(
        perturbed_results['trajectories'], 
        title="扰动模拟轨道 (0.05%)", 
        save_path="perturbed_orbits.png"
    )
    
    # 绘制能量守恒图
    plot_energy_conservation(
        standard_results['energies'],
        standard_results['energies'][0],
        title="标准模拟能量守恒",
        save_path="standard_energy.png"
    )
    
    # 绘制轨道要素演化图
    plot_orbit_elements(
        standard_results['orbit_elements'],
        title="标准模拟轨道要素演化",
        save_path="standard_orbit_elements.png"
    )
    
    # 生成动画
    generate_animation(
        standard_results['trajectories'],
        title="标准模拟轨道演化",
        filename="standard_animation.gif"
    )
    
    print("所有结果已保存至results目录")

if __name__ == "__main__":
    main()
