import os
import time

def create_results_directory():
    """创建结果保存目录"""
    if not os.path.exists('results'):
        os.makedirs('results')
    os.chdir('results')

def timer_decorator(func):
    """计时装饰器"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{func.__name__} 运行时间: {end_time - start_time:.2f} 秒")
        return result
    return wrapper
