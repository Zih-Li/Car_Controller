# 从png提取地图。
# 该代码最后会在main exp 3.py中使用。

import numpy as np
from PIL import Image, ImageDraw
import math
import matplotlib.pyplot as plt
import os
from skimage.draw import polygon as sk_polygon # 用于绘制车辆多边形
Image.MAX_IMAGE_PIXELS = None
map_path = "map.png"  # 地图图片文件路径

class map_and_obstacle:
    def __init__(self, resolution_val=1600, world_size_val=(160, 160), green_threshold_sq_val=30000):
        # 参数：
        # 分辨率：初始为1600x1600
        self.resolution = (resolution_val, resolution_val)  # 假设地图是正方形
        self.world_size = world_size_val  # 世界的（宽度，高度），单位为米/单位，例如 (160, 160)
        
        # 计算比例因子
        self.scale_x = self.resolution[0] / self.world_size[0] if self.world_size[0] > 0 else 1
        self.scale_y = self.resolution[1] / self.world_size[1] if self.world_size[1] > 0 else 1

        # 障碍物列表：初始为空
        # 此列表将存储车辆数据。期望的字典格式：
        # {'x': 世界坐标x, 'y': 世界坐标y, 'yaw': 弧度, 'length': 世界单位长度, 'width': 世界单位宽度}
        self.obstacle_vehicles = []
            # 该部分会实时接收全局车辆信息，全局信息格式参考my_udp.py，中的UDPClient类。
            # 包括全局车辆的坐标，速度，朝向等。
        
        # 地图数组 (来自PNG)
        self.static_map_array = None
        self.map_image_path = None # 存储路径以便重新加载

        # 地图及障碍物数组（基于分辨率，从sample_from_map()获取 -> 即 map_and_obstacle_to_array()）
        # 这将是组合后的地图。不作为成员变量存储，而是由 map_and_obstacle_to_array() 返回

        # PNG地图中不可通行区域的颜色定义
        self.nontraversable_color_rgb = (0, 255, 0) # 纯绿色
        # 用于判断颜色是否为“绿色”的欧氏距离平方阈值
        # 默认值允许每个通道平均偏差 sqrt(7500/3) = 50
        self.green_color_distance_threshold_sq = green_threshold_sq_val

    def _world_to_grid(self, x_w, y_w):
        """
        将世界坐标转换为栅格/数组坐标。
        世界坐标：原点 (0,0) 在左下角，X 轴向右增加，Y 轴向上增加。
        栅格坐标：原点 (0,0) 在左上角（标准数组索引），列索引向右增加，行索引向下增加。
        返回 (列索引, 行索引)
        """
        # 列索引对应世界 X，行索引对应世界 Y
        # 世界 X=0 (左边界) -> 列索引 = 0
        # 世界 X=world_size_x (右边界) -> 列索引 = resolution_x - 1
        col_idx_float = x_w * self.scale_x
        
        # 世界 Y=0 (下边界) -> 行索引 = resolution_y - 1
        # 世界 Y=world_size_y (上边界) -> 行索引 = 0
        row_idx_float = (self.world_size[1] - y_w) * self.scale_y
        
        # 转换为整数并裁剪到数组边界内
        col_idx = np.clip(int(round(col_idx_float)), 0, self.resolution[0] - 1)
        row_idx = np.clip(int(round(row_idx_float)), 0, self.resolution[1] - 1)
        
        return col_idx, row_idx


    def load_map_from_png(self, png_filepath):
        """
        从PNG文件加载地图。
        不可通行区域由 self.nontraversable_color_rgb 标记。
        更新 self.static_map_array。
        黄色像素赋值为0.5，白色像素赋值为0.2。
        """
        self.map_image_path = png_filepath
        try:
            raw_image = Image.open(png_filepath).convert("RGB")
            
            # 如果需要，调整大小以匹配 self.resolution
            if raw_image.size != self.resolution:
                raw_image = raw_image.resize(self.resolution, Image.NEAREST) # 最近邻插值
            
            img_array = np.array(raw_image) # 形状 (高度, 宽度, 3)
            
            # 识别不可通行（绿色）的像素
            target_color_rgb = np.array(self.nontraversable_color_rgb)
            squared_differences = (img_array.astype(np.int32) - target_color_rgb.astype(np.int32))**2
            sum_squared_differences = np.sum(squared_differences, axis=-1)
            nontraversable_pixels = sum_squared_differences < self.green_color_distance_threshold_sq

            # 检测黄色 (255,255,0)
            yellow_rgb = np.array([255, 255, 0])
            yellow_diff = (img_array.astype(np.int32) - yellow_rgb.astype(np.int32))**2
            yellow_dist = np.sum(yellow_diff, axis=-1)
            yellow_pixels = yellow_dist < 2000  # 阈值可根据实际情况调整

            self.static_map_array = np.zeros(self.resolution, dtype=np.float32) # 0 代表可通行
            self.static_map_array[nontraversable_pixels] = 1 # 1 代表不可通行
            self.static_map_array[yellow_pixels] = 1        # 黄色也视为障碍，赋值为1

            # PNG 图像被假定为这样一种方向：其视觉上的左上角
            # (像素 0,0) 对应于世界坐标 (X=0, Y=world_size[1]) (世界区域的左上角),
            # 其视觉上的右下角对应于世界坐标 (X=world_size[0], Y=0) (世界区域的右下角)。
            # self.static_map_array 存储的地图中 (row=0, col=0) 是左上角的栅格单元，
            # 与 PNG 图像的像素数组方向一致。
            return True
        except FileNotFoundError:
            print(f"错误：在 {png_filepath} 未找到地图 PNG 文件")
            self.static_map_array = None
            return False
        except Exception as e:
            print(f"错误：加载或处理地图 PNG 时出错: {e}")
            self.static_map_array = None
            return False

    def update_vehicle_obstacles(self, vehicles_data_list):
        """
        更新车辆障碍物列表。
        vehicles_data_list 中的每一项都应该是一个字典：
        {'x': 世界坐标x, 'y': 世界坐标y, 'yaw': 弧度, 'length': 世界单位长度, 'width': 世界单位宽度}
        Yaw (偏航角) 应为弧度。
        """
        self.obstacle_vehicles = vehicles_data_list

    def add_obstacle(self):
        # 任务：
        # 在采样的地图中，根据障碍物（车辆）坐标及其大小，生成障碍物数组。
        # 障碍物数组的大小与采样地图分辨率一致，0代表无障碍物，1代表有障碍物。
        # 根据车辆的坐标，大小，朝向，将相应位置的障碍物数组置为1。
        # 此处车辆坐标为车辆中心点坐标，大小为车辆的宽度和长度。
        """
        生成一个仅表示动态车辆障碍物的障碍物数组。
        该数组的大小与地图分辨率相同 (例如, 1600x1600)。
        0 表示该栅格单元没有车辆障碍物，1 表示存在车辆障碍物。
        使用 self.obstacle_vehicles。
        """
        vehicle_obstacle_layer = np.zeros(self.resolution, dtype=np.uint8)

        for vehicle in self.obstacle_vehicles:
            center_x_w, center_y_w = vehicle['x'], vehicle['y']
            yaw_rad = vehicle['yaw']  # 期望单位为弧度
            length_w, width_w = vehicle['length'], vehicle['width']

            # 半长和半宽
            hl, hw = length_w / 2.0, width_w / 2.0

            # 在车辆局部坐标系中定义角点 (x向前, y向左)
            # (可能是 +x 向前, +y 向左, 或其他约定; 确保一致性)
            # 假设 +x 是长度轴, +y 是从中心开始的宽度轴
            corners_vehicle_frame = [
                (hl, -hw),  # 相对于车辆中心的前右角点
                (hl, hw),   # 前左角点
                (-hl, hw),  # 后左角点
                (-hl, -hw)  # 后右角点
            ]

            rotated_corners_grid_coords = []
            for cvx, cvy in corners_vehicle_frame:
                # 按偏航角旋转角点
                x_rot_world_offset = cvx * math.cos(yaw_rad) - cvy * math.sin(yaw_rad)
                y_rot_world_offset = cvx * math.sin(yaw_rad) + cvy * math.cos(yaw_rad)
                
                # 平移到世界坐标
                x_world_corner = center_x_w + x_rot_world_offset
                y_world_corner = center_y_w + y_rot_world_offset
                
                # 将世界坐标角点转换为栅格坐标 (列索引, 行索引)
                g_col, g_row = self._world_to_grid(x_world_corner, y_world_corner)
                rotated_corners_grid_coords.append((g_row, g_col)) # skimage.draw.polygon 需要 (行, 列)

            if rotated_corners_grid_coords:
                # 提取 skimage.draw.polygon 所需的行和列坐标
                row_coords = np.array([p[0] for p in rotated_corners_grid_coords])
                col_coords = np.array([p[1] for p in rotated_corners_grid_coords])
                
                # 获取多边形内部的像素
                rr, cc = sk_polygon(row_coords, col_coords, shape=vehicle_obstacle_layer.shape)
                
                # 在车辆层中将这些像素标记为障碍物
                vehicle_obstacle_layer[rr, cc] = 1
        
        return vehicle_obstacle_layer

    def map_and_obstacle_to_array(self):
        # 信息：
        # map图像中，不可行区域为绿色，其他为可行区。
        # map的坐标为左下角为原点(0, 0)，右上角为(160, 160)。
        
        # 任务：
        # 请以1600x1600的分辨率，采样出地图。 (已由 load_map_from_png 完成)
        # 返回一个1600x1600的numpy数组，表示采样后的地图。
        # 该数组中，0代表可行区，1代表不可行区，2代表黄线，3代表白线
        # 加入障碍物。 (将静态地图与车辆障碍物合并)

        if self.static_map_array is None:
            if self.map_image_path:
                print("静态地图未加载。尝试从存储的路径加载...")
                if not self.load_map_from_png(self.map_image_path):
                    print("加载静态地图失败。无法生成组合地图。")
                    return None
            else:
                print("错误：静态地图未加载。请先调用 load_map_from_png()。")
                return None
        
        # 生成车辆障碍物层
        vehicle_obstacle_layer = self.add_obstacle()
        
        static_map = self.static_map_array.copy()

        combined_map = np.zeros_like(static_map, dtype=np.uint8)

        combined_map[(static_map == 1) | (vehicle_obstacle_layer == 1)] = 1

        # combined_map[(combined_map == 0) & (static_map == 2)] = 2

        # combined_map[(combined_map == 0) & (static_map == 3)] = 3

        # 其余为0（可通行）
        return combined_map

    def print_map(self, map_array_to_print=None, title="地图显示"):
        """
        打印采样后的地图/障碍物数组，支持多种颜色显示：
        1.0=绿色（不可通行），0.5=黄色，0.2=白色，0=可通行（黑色），车辆障碍物为红色（如合并后为1）。
        """
        import matplotlib.colors as mcolors

        if map_array_to_print is None:
            print("未提供要打印的地图数组。")
            return

        plt.figure(figsize=(10, 10))
        extent_params = [0, self.world_size[0], 0, self.world_size[1]]

        # 自定义颜色映射
        cmap = mcolors.ListedColormap([
            "#FFFFFF", 
            "#070707",
            "#e0e03e",
            "#C1B5B5",
        ])
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        plt.imshow(map_array_to_print, cmap=cmap, norm=norm, origin='upper', extent=extent_params)
        plt.xlabel(f"世界 X坐标 (左: 0, 右: {self.world_size[0]})")
        plt.ylabel(f"世界 Y坐标 (下: 0, 上: {self.world_size[1]})")
        plt.title(title)
        cbar = plt.colorbar(ticks=[0, 1, 2, 3])
        cbar.ax.set_yticklabels(['可通行(黑)', '障碍/绿色', '黄线', '白线'])
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        plt.show()


if __name__ == "__main__":
    print("运行 map_and_obstacle.py 自测试...")

    
    
    # 为测试创建一个虚拟的 map_image.png
    try:
        # 创建一个小的虚拟 PNG (例如, 160x160)
        # load_map_from_png 会将其调整为 self.resolution
        test_img_size = (160, 160) # 在这个虚拟示例中，为简单起见，与 world_size 匹配
        img = Image.new("RGB", test_img_size, "white") # 初始全部可通行
        draw = ImageDraw.Draw(img)
        
        # 添加一个绿色 (不可通行) 矩形。
        # PIL 的 draw 使用 (左, 上, 右, 下) 像素坐标。
        # 图像像素 (0,0) 是左上角。
        # 如果这个绿色框在 160x160 图像中从像素 (20,20) 到 (60,60)：
        # 这对应于：
        #   图像行 20-60, 图像列 20-60。
        #   世界 X 从 20 到 60 (如果比例为 1)。
        #   世界 Y 从 (160-60)=100 到 (160-20)=140 (如果比例为 1)。
        draw.rectangle([20, 20, 60, 60], fill=(0,255,0)) # 绿色
        
        dummy_map_path = "dummy_map.png"

        img.save(dummy_map_path)
        print(f"已在 {dummy_map_path} 创建虚拟地图")

        # 初始化 map_and_obstacle 实例
        # 使用 800x800 分辨率以加快测试，默认为 1600x1600
        mapper = map_and_obstacle(resolution_val=1000, world_size_val=(160, 160))

        if os.path.exists(map_path): # 检查实际地图文件是否存在
            dummy_map_path = map_path # 如果存在，则使用实际地图进行测试
        
        # 1. 从 PNG 加载地图
        if mapper.load_map_from_png(dummy_map_path):
            print("静态地图加载成功。")
            # 可选：打印静态地图
            # mapper.print_map(mapper.static_map_array, title="静态地图 (来自 PNG)")

            # 2. 定义一些车辆障碍物
            # VehicleData 格式: {'x': 世界坐标, 'y': 世界坐标, 'yaw': 弧度, 'length': 世界单位, 'width': 世界单位}
            # Yaw=0 表示车辆的长度轴与世界 X 轴正方向对齐 (指向右)。
            # Yaw=pi/2 表示车辆的长度轴与世界 Y 轴正方向对齐 (指向上)。
            vehicles = [
                {'name': 'v1', 'x': 0, 'y': 0, 'yaw': 0, 'length': 10, 'width': 5},
                {'name': 'v2', 'x': 40, 'y': 120, 'yaw': math.pi/4, 'length': 12, 'width': 6}, # 旋转 (从+X逆时针旋转45度)
                {'name': 'v3', 'x': 130, 'y': 30, 'yaw': -math.pi/2, 'length': 15, 'width': 4} # 指向“下” (沿世界Y轴负方向)
            ]
            mapper.update_vehicle_obstacles(vehicles)
            print(f"已使用 {len(vehicles)} 辆车更新车辆障碍物。")

            # 性能测试：测量读取车辆位置并生成动态地图数组的频率（Hz）
            import time
            iterations = 100
            start_time = time.time()
            for _ in range(iterations):
                mapper.update_vehicle_obstacles(vehicles)
                _ = mapper.map_and_obstacle_to_array()
            elapsed = time.time() - start_time
            freq = iterations / elapsed if elapsed > 0 else float('inf')
            print(f"性能测试：{iterations} 次更新+生成耗时 {elapsed:.4f} s, 频率 {freq:.2f} Hz")

            # 3. 生成带障碍物的地图
            combined_map = mapper.map_and_obstacle_to_array()
            if combined_map is not None:
                print("已生成带障碍物的组合地图。")
                mapper.print_map(combined_map, title=f"组合地图 ({mapper.resolution[0]}x{mapper.resolution[1]})")
            else:
                print("生成组合地图失败。")
        else:
            print("加载地图失败。")

    except ImportError as e:
        print(f"ImportError: {e}。请确保已安装 Pillow, NumPy, Matplotlib, 和 scikit-image。")
    except Exception as e:
        print(f"测试脚本中发生错误: {e}")
        import traceback
        traceback.print_exc()

