import cv2
import json
import numpy as np
import argparse
import os

def load_map_image(map_path):
    """加载地图图像"""
    if not os.path.exists(map_path):
        raise FileNotFoundError(f"地图文件不存在: {map_path}")
    
    img = cv2.imread(map_path)
    if img is None:
        raise ValueError(f"无法读取地图文件: {map_path}")
    
    return img

def load_trajectory(traj_path):
    """加载轨迹JSON文件"""
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"轨迹文件不存在: {traj_path}")
    
    with open(traj_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def world_to_image_coords(x, y, img_height, world_width=160, world_height=160):
    """
    将世界坐标转换为图像坐标
    世界坐标系：左下角为原点(0,0)，右上角为(160,160)
    图像坐标系：左上角为原点(0,0)，右下角为(img_width, img_height)
    """
    # 计算图像宽度和高度
    img_width = img_height  # 假设图像是正方形，如果不是需要调整
    
    # 世界坐标到图像坐标的转换
    img_x = int((x / world_width) * img_width)
    img_y = int(img_height - (y / world_height) * img_height)  # y轴翻转
    
    return img_x, img_y

def draw_trajectory_points(img, trajectory_data, color, point_size=20, draw_lines=False):
    """在图像上绘制轨迹点"""
    img_height, img_width = img.shape[:2]
    
    # 处理不同格式的轨迹数据
    if isinstance(trajectory_data, list):
        points = trajectory_data
    elif isinstance(trajectory_data, dict):
        # 如果是字典，尝试找到坐标数据
        if 'trajectory' in trajectory_data:
            points = trajectory_data['trajectory']
        elif 'points' in trajectory_data:
            points = trajectory_data['points']
        elif 'path' in trajectory_data:
            points = trajectory_data['path']
        else:
            # 假设字典的值就是点列表
            points = list(trajectory_data.values())[0] if trajectory_data else []
    else:
        points = []
    
    # 存储转换后的图像坐标点
    img_points = []
    
    # 绘制点并收集坐标
    for point in points:
        if isinstance(point, dict):
            x = point.get('x', 0)
            y = point.get('y', 0)
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            x, y = point[0], point[1]
        else:
            continue
        
        # 转换坐标
        img_x, img_y = world_to_image_coords(x, y, img_height)
        
        # 确保坐标在图像范围内
        if 0 <= img_x < img_width and 0 <= img_y < img_height:
            cv2.circle(img, (img_x, img_y), point_size, color, -1)
            img_points.append((img_x, img_y))
    
    # 如果需要绘制线条，连接相邻的点
    if draw_lines and len(img_points) > 1:
        for i in range(len(img_points) - 1):
            cv2.line(img, img_points[i], img_points[i + 1], color, 20)

def main():
    parser = argparse.ArgumentParser(description='在地图上绘制轨迹')
    parser.add_argument('--map', default='map.png', help='地图文件路径')
    parser.add_argument('--recorded', default='recorded_traj.json', help='记录轨迹文件路径')
    parser.add_argument('--reference', default='ref_route.json', help='参考路径文件路径')
    parser.add_argument('--with_ori', action='store_true', help='是否绘制参考路径')
    parser.add_argument('--line', action='store_true', help='是否连点成线')
    parser.add_argument('--output', default='trajectory_visualization.png', help='输出图像路径')
    args = parser.parse_args()

    try:
        # 加载地图图像
        print(f"加载地图: {args.map}")
        img = load_map_image(args.map)
        
        # 加载记录的轨迹
        print(f"加载记录轨迹: {args.recorded}")
        recorded_traj = load_trajectory(args.recorded)
        
        # 在图像上绘制记录的轨迹（蓝色）

        
        # 如果指定了--with_ori参数，则绘制参考路径
        if args.with_ori:
            print(f"加载参考路径: {args.reference}")
            if args.reference == 'ref_route.json':
                ref_route = json.load(open("ref_route.json", "r"))
                static_traj = []
                x = ref_route["X"]
                y = ref_route["Y"]
                for xi, yi in zip(x, y):
                    static_traj.append([xi*10, yi*10])
                ref_route = static_traj

            else:
                ref_route = load_trajectory(args.reference)

            # 在图像上绘制参考路径（白色）
            print("绘制参考路径...")
            draw_trajectory_points(img, ref_route, color=(255, 255, 255), point_size=5, draw_lines=args.line)  # 白色
        print("绘制记录轨迹...")
        draw_trajectory_points(img, recorded_traj, color=(255, 0, 0), point_size=5, draw_lines=args.line)  # 蓝色

        # 保存结果图像
        print(f"保存结果图像: {args.output}")
        cv2.imwrite(args.output, img)
        
        print("完成！")
        
        # 显示图像（可选）
        cv2.imshow('Trajectory Visualization', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())