import json
import math
import matplotlib.pyplot as plt

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def read_json_points(filename):
    """读取json文件并返回点坐标列表"""
    with open(filename, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # JSON格式为 [[x1, y1], [x2, y2], ...]
    points = [(point[0], point[1]) for point in data]
    return points

def calculate_distance(p1, p2):
    """计算两点之间的距离"""
    return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def remove_duplicates(points):
    """去除相同的点"""
    unique_points = []
    for point in points:
        if point not in unique_points:
            unique_points.append(point)
    return unique_points

def filter_trajectory(points):
    """过滤轨迹：点与点距离不超过1，轨迹至少10个点"""
    if len(points) < 10:
        return []
    
    filtered_points = [points[0]]
    
    for i in range(1, len(points)):
        distance = calculate_distance(points[i-1], points[i])
        if distance <= 1:
            filtered_points.append(points[i])
        else:
            # 距离超过1，结束当前轨迹段
            if len(filtered_points) >= 10:
                return filtered_points
            else:
                filtered_points = [points[i]]
    
    return filtered_points if len(filtered_points) >= 10 else []

def print_trajectory(points):
    """打印轨迹坐标点"""
    print(f"轨迹包含 {len(points)} 个点:")
    for i, (x, y) in enumerate(points):
        print(f"点 {i+1}: ({x:.2f}, {y:.2f})")

def plot_trajectory(points):
    """绘制轨迹图"""
    if not points:
        print("没有有效轨迹可绘制")
        return
    
    x_coords = [p[0] for p in points]
    y_coords = [p[1] for p in points]
    
    plt.figure(figsize=(10, 8))
    plt.plot(x_coords, y_coords, 'b-o', markersize=3)
    
    # 设置等间隔的刻度
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    
    # 计算合适的间隔
    x_range = x_max - x_min
    y_range = y_max - y_min
    
    x_interval = max(1, round(x_range / 10))
    y_interval = max(1, round(y_range / 10))
    
    # 设置刻度
    x_ticks = range(int(x_min), int(x_max) + 1, x_interval)
    y_ticks = range(int(y_min), int(y_max) + 1, y_interval)
    
    plt.xticks(x_ticks)
    plt.yticks(y_ticks)
    
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')
    plt.title('车辆轨迹')
    plt.grid(True)
    plt.axis('equal')  # 确保x和y轴比例相等
    plt.show()

# 主程序
if __name__ == "__main__":
    try:
        # 读取JSON文件
        filename = "trajectory.json"  # 替换为实际文件名
        points = read_json_points(filename)
        
        # 去除重复点
        points = remove_duplicates(points)
        
        # 过滤轨迹
        valid_trajectory = filter_trajectory(points)
        
        if valid_trajectory:
            # 打印轨迹
            print_trajectory(valid_trajectory)
            
            # 绘制轨迹
            plot_trajectory(valid_trajectory)
        else:
            print("没有找到符合条件的完整轨迹")
            
    except FileNotFoundError:
        print("JSON文件未找到，请检查文件路径")
    except Exception as e:
        print(f"处理过程中出现错误: {e}")