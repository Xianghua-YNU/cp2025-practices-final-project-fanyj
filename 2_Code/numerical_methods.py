"""Verlet算法实现：求解三体问题的数值积分器"""
import numpy as np

class VerletIntegrator:
    """基于Velocity Verlet算法的多体系统积分器"""
    def __init__(self, G):
        self.G = G  # 万有引力常数
        self.bodies = []  # 存储天体信息：(质量, 位置, 速度)
        self.positions = []  # 记录历史位置
        self.velocities = []  # 记录历史速度
        self.times = []  # 记录时间
    
    def add_body(self, mass, position, velocity):
        """添加天体到系统"""
        self.bodies.append({
            "mass": mass,
            "position": np.array(position, dtype=float),
            "velocity": np.array(velocity, dtype=float),
            "acceleration": np.zeros(3)
        })
    
    def reset(self):
        """重置积分器状态"""
        self.bodies = []
        self.positions = []
        self.velocities = []
        self.times = []
    
    def calculate_accelerations(self):
        """计算所有天体的加速度（基于当前位置）"""
        for i, body_i in enumerate(self.bodies):
            acc = np.zeros(3)
            for j, body_j in enumerate(self.bodies):
                if i == j:
                    continue
                # 万有引力加速度：F = G*m_i*m_j/r^2 * r_hat
                r_ij = body_j["position"] - body_i["position"]
                r = np.linalg.norm(r_ij)
                if r < 1e-10:  # 避免零距离
                    continue
                acc += self.G * body_j["mass"] * r_ij / (r ** 3)
            body_i["acceleration"] = acc
    
    def simulate(self, total_days, dt, output_interval):
        """执行多体系统模拟"""
        total_steps = int(total_days * 86400 / dt)  # 总步数（秒转换）
        output_steps = int(output_interval * 86400 / dt)  # 输出间隔步数
        
        # 初始化记录数组
        num_bodies = len(self.bodies)
        self.positions = np.zeros((total_steps//output_steps + 1, num_bodies, 3))
        self.velocities = np.zeros((total_steps//output_steps + 1, num_bodies, 3))
        self.times = np.zeros(total_steps//output_steps + 1)
        
        # 保存初始状态
        for i, body in enumerate(self.bodies):
            self.positions[0, i] = body["position"]
            self.velocities[0, i] = body["velocity"]
        self.times[0] = 0
        
        # Velocity Verlet 迭代
        step = 0
        output_idx = 0
        for t in range(total_steps):
            current_time = t * dt
            
            # 1. 计算加速度
            self.calculate_accelerations()
            
            # 2. 更新位置
            for body in self.bodies:
                body["position"] += body["velocity"] * dt + 0.5 * body["acceleration"] * dt**2
            
            # 3. 再次计算加速度（新位置）
            self.calculate_accelerations()
            
            # 4. 更新速度
            for body in self.bodies:
                body["velocity"] += 0.5 * (body["acceleration"] + body["acceleration"]) * dt
            
            # 5. 按间隔输出
            if t % output_steps == 0:
                for i, body in enumerate(self.bodies):
                    self.positions[output_idx, i] = body["position"]
                    self.velocities[output_idx, i] = body["velocity"]
                self.times[output_idx] = current_time / 86400  # 转换为天
                output_idx += 1
                step += 1
                print(f"模拟进度: {step}/{total_steps//output_steps}")
        
        return self.times[:output_idx], self.positions[:output_idx], self.velocities[:output_idx]
