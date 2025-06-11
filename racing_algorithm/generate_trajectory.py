# 该generate_trajectory.py文件封装实时轨迹生成内容
# 由于该地图且该比赛不设终点，我们的实时轨迹算法为搜索一条有限长度路径，该长度可调整
# 该文件最后被car_control_agent.py调用
# 特别注意地图数组和世界坐标系下的转换，该文件中的地图数组具体实现和车辆状态转换等内容在map_and_obstacle.py中实现
# 车长5m，车宽2.2m，轴距2.86m，轮距1.69m，最大转向角35度

import numpy as np
import heapq
import math
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.interpolate import make_interp_spline
import time
import numba
from numba import typed
from concurrent.futures import ThreadPoolExecutor, as_completed

# 假设 map_and_obstacle.py 在同一目录或可访问的路径中
# 它提供了 map_and_obstacle 类用于坐标变换和地图加载
try:
    from racing_algorithm.map_and_obstacle import map_and_obstacle
except ImportError:
    print("错误：无法导入 'map_and_obstacle'。请确保 map_and_obstacle.py 文件在同一目录下。")
    exit()

# ==============================================================================
# Numba JIT 加速的辅助函数 (在模块级别定义以便Numba编译)
# ==============================================================================


@numba.jit(nopython=True, cache=True)
def _get_min_dist_to_path_numba(point_x, point_y, path_arr):
    """[Numba加速] 计算一个点到一条路径(点序列)的最短距离"""
    if path_arr.shape[0] == 0:
        return 1e9 # 如果路径为空，返回一个很大的距离
    
    min_dist_sq = 1e18 # 使用距离的平方进行比较，避免开方
    for i in range(path_arr.shape[0]):
        dx = point_x - path_arr[i, 0]
        dy = point_y - path_arr[i, 1]
        dist_sq = dx**2 + dy**2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
    return math.sqrt(min_dist_sq)



@numba.jit(nopython=True, cache=True)
def _grid_to_world_numba(col_idx, row_idx, scale_x, scale_y, world_height):
    """
    [Numba加速] 将栅格坐标(列,行)转换为世界坐标(x,y)。
    """
    x_w = col_idx / scale_x
    y_w = world_height - (row_idx / scale_y)
    return x_w, y_w

@numba.jit(nopython=True, cache=True)
def _world_to_grid_numba(x_w, y_w, scale_x, scale_y, world_height, map_rows, map_cols):
    """
    [Numba加速] 将世界坐标(x,y)转换为栅格坐标(列,行)。
    """
    col_idx = x_w * scale_x
    row_idx = (world_height - y_w) * scale_y
    
    # 将坐标限制在地图边界内，防止因浮点误差导致的越界
    final_col = max(0, min(int(round(col_idx)), map_cols - 1))
    final_row = max(0, min(int(round(row_idx)), map_rows - 1))
    return final_col, final_row

@numba.jit(nopython=True, cache=True)
def _bresenham_line_check_numba(r0, c0, r1, c1, swelled_map):
    """
    [Numba加速] 使用Bresenham算法检查两点间的直线路径是否与障碍物碰撞。
    只检测障碍物（值为1），忽略道路线（值为2或3）。
    
    返回:
        bool: 如果路径上有障碍物或越界，则返回True，否则返回False。
    """
    r0, c0 = int(r0), int(c0)
    r1, c1 = int(r1), int(c1)
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    map_rows, map_cols = swelled_map.shape
    curr_r, curr_c = r0, c0
    max_iter = map_rows + map_cols  # 安全迭代上限

    for _ in range(max_iter):
        if not (0 <= curr_r < map_rows and 0 <= curr_c < map_cols):
            return True  # 越界视为碰撞

        if swelled_map[curr_r, curr_c] == 1:
            return True  # 撞到障碍物

        if curr_r == r1 and curr_c == c1:
            break  # 到达终点

        e2 = 2 * err
        if e2 >= -dr:
            err -= dr
            curr_c += sc
        if e2 <= dc:
            err += dc
            curr_r += sr
            
    return False # 路径安全

@numba.jit(nopython=True, cache=True)
def _process_steering_angle_numba(
    current_col_grid, current_row_grid, current_yaw, steer_angle, current_steer,
    forward_flag, lookahead_dist,
    scale_x, scale_y, world_height, map_rows_px, map_cols_px,
    swelled_map_arr, dist_transform_arr,
    cost_w_steer, cost_w_obs, cost_w_reverse,
    wheelbase,
    prev_traj, cost_w_path_deviation, cost_w_steer_rate, cost_w_curvature
):
    """
    [Numba加速] 处理单个转向角度，执行运动学计算、碰撞检测和成本计算。
    这是启发式搜索中的核心扩展步骤。

    返回:
        (is_valid, next_col, next_row, next_yaw, edge_cost)
        is_valid (bool): 如果路径有效（无碰撞、未越界）则为True。
    """
    # 1. 确定运动方向
    direction = 1.0 if forward_flag else -1.0

    # 2. 获取当前世界坐标
    current_world_x, current_world_y = _grid_to_world_numba(
        current_col_grid, current_row_grid, scale_x, scale_y, world_height
    )

    # 3. 使用运动学自行车模型计算下一位置和朝向
    distance = direction * lookahead_dist

    # 如果转向角非常小，视为直线运动以避免数值问题
    if abs(steer_angle) < 1e-6:
        next_world_x = current_world_x + distance * math.cos(current_yaw)
        next_world_y = current_world_y + distance * math.sin(current_yaw)
        next_yaw = current_yaw
    else:
        # 计算转弯半径
        turn_radius = wheelbase / math.tan(steer_angle)
        
        # 转过的角度 (rad)
        beta = distance / turn_radius
        
        # 新的朝向角
        next_yaw = current_yaw + beta
        next_yaw = (next_yaw + math.pi) % (2 * math.pi) - math.pi  # 标准化到[-pi, pi]
        
        # 计算旋转中心
        rotation_center_x = current_world_x - turn_radius * math.sin(current_yaw)
        rotation_center_y = current_world_y + turn_radius * math.cos(current_yaw)
        
        # 计算新位置
        next_world_x = rotation_center_x + turn_radius * math.sin(next_yaw)
        next_world_y = rotation_center_y - turn_radius * math.cos(next_yaw)

    # 4. 转换回栅格坐标
    next_col_grid, next_row_grid = _world_to_grid_numba(
        next_world_x, next_world_y, scale_x, scale_y, world_height, map_rows_px, map_cols_px
    )
    
    # 5. 路径碰撞检测 (使用Bresenham算法)
    if _bresenham_line_check_numba(current_row_grid, current_col_grid, 
                                 next_row_grid, next_col_grid, swelled_map_arr):
        return False, -1, -1, 0.0, 0.0

    # 6. 计算综合成本
    # 转向成本
    steer_cost = cost_w_steer * abs(steer_angle)

    # 急转向速率成本
    steer_rate = (steer_angle - current_steer) / lookahead_dist
    steer_rate_cost = cost_w_steer_rate * abs(steer_rate)

    # 计算曲率成本
    curvature = abs(steer_angle) / wheelbase
    curvature_cost = cost_w_curvature * curvature
    # 障碍物成本 (基于到最近障碍物的距离)
    dist_to_obs = dist_transform_arr[next_row_grid, next_col_grid]
    obs_cost = cost_w_obs / (dist_to_obs + 1e-6) # 加一个小数防止除以零
    
    # 倒车成本
    reverse_cost = 0.0 if forward_flag else cost_w_reverse

    path_dev_cost = 0.0
    if prev_traj.shape[0] > 0:
        dist_to_prev_path = _get_min_dist_to_path_numba(next_world_x, next_world_y, prev_traj)
        path_dev_cost = cost_w_path_deviation * dist_to_prev_path

    edge_cost = steer_cost + obs_cost + reverse_cost + path_dev_cost + curvature_cost + steer_rate_cost
    return True, next_col_grid, next_row_grid, next_yaw, edge_cost

# ==============================================================================
# 轨迹生成主类
# ==============================================================================

class GenerateTrajectory:
    """
    实时轨迹生成器。
    
    该类实现了一个混合式实时轨迹生成系统，其主要特点和功能包括：
    1.  **障碍物膨胀**：根据车辆尺寸膨胀地图中的障碍物，为规划留出安全裕度。
    2.  **直线捷径优先**：首先尝试生成一条无碰撞的直线路径。这是一种高效的启发式策略，
        在开阔区域能极大地加速规划过程。
    3.  **启发式搜索**：如果直线路径不可行（例如需要避障），则启动基于运动学模型的启发式搜索算法
        （类似Hybrid A*），在(x, y, yaw)状态空间中寻找一条可行路径。
    4.  **路径平滑**：使用B样条对生成的离散路径点进行平滑处理，使其更适合车辆跟踪。
    
    性能优化措施：
    -   关键计算（坐标转换、碰撞检测、运动学模型）使用 `Numba` 进行了即时编译（JIT）加速。
    -   在启发式搜索的节点扩展阶段，使用 `ThreadPoolExecutor` 并行处理不同的转向角采样，
        充分利用多核CPU资源。
    """
    def __init__(self, map_obj: map_and_obstacle, vehicle_params: dict = None, plan_params: dict = None):
        """
        初始化轨迹生成器。

        Args:
            map_obj (map_and_obstacle): 地图处理器类的实例。
            vehicle_params (dict, optional): 车辆物理参数。
                - 'width' (float): 车辆宽度(米)，用于膨胀计算。默认 4.0。
                - 'max_steer' (float): 最大转向角(弧度)。默认 pi/6。
                - 'wheelbase' (float): 车辆轴距(米)。默认 2.86。
            plan_params (dict, optional): 路径规划参数。
                - 'target_path_len' (float): 目标路径长度(米)。默认 30.0。
                - 'lookahead_dist' (float): 搜索时的前瞻距离/步长(米)。默认 8.0。
                - 'num_steer_samples' (int): 转向角采样数量。默认 7。
                - 'cost_w_steer' (float): 转向代价权重。默认 5.0。
                - 'cost_w_obs' (float): 障碍物代价权重。默认 10.0。
                - 'cost_w_reverse' (float): 倒车代价权重。默认 20.0。
        """
        self.map_obj = map_obj
        
        # 合并用户定义参数和默认参数
        default_vehicle_params = {
            'width': 4.0,
            'max_steer': np.pi / 6,
            'wheelbase': 2.86
        }
        self.vehicle_params = {**default_vehicle_params, **(vehicle_params or {})}
        
        default_plan_params = {
            'target_path_len': 30.0,
            'lookahead_dist': 8.0,
            'num_steer_samples': 5,
            'cost_w_steer': 5.0,
            'cost_w_obs': 10.0,
            'cost_w_reverse': 20.0,
            'cost_w_path_deviation': 20.0,
            'cost_w_steer_rate': 15.0,
            'cost_w_curvature': 10.0
        }
        self.plan_params = {**default_plan_params, **(plan_params or {})}
        self.wheelbase = self.vehicle_params['wheelbase']
        

        # 预计算转向角采样，确保采样数为奇数以包含0度转向
        max_steer = self.vehicle_params['max_steer']
        num_samples = self.plan_params['num_steer_samples']
        if num_samples % 2 == 0:
            num_samples += 1
        self.steer_samples = np.linspace(-max_steer, max_steer, num_samples)
        self.previous_trajectory = None
    def _grid_to_world(self, col_idx, row_idx):
        """将栅格坐标(列,行)转换为世界坐标(x,y)。"""
        return _grid_to_world_numba(
            col_idx, row_idx, self.map_obj.scale_x, self.map_obj.scale_y, self.map_obj.world_size[1]
        )

    def get_and_swell_map(self, map_array: np.ndarray) -> np.ndarray:
        """
        根据车辆宽度膨胀地图中的障碍物。
        此方法只膨胀值为1的障碍物，而保持道路线（值2和3）不变。

        Args:
            map_array (np.ndarray): 原始二维地图数组 (0=可行, 1=障碍, 2/3=线)。

        Returns:
            np.ndarray: 膨胀处理后的地图数组。
        """
        swell_radius_m = self.vehicle_params['width'] / 2.0
        avg_scale = (self.map_obj.scale_x + self.map_obj.scale_y) / 2.0
        swell_radius_px = int(round(swell_radius_m * avg_scale))

        if swell_radius_px <= 0:
            return map_array.copy()

        # 创建一个掩码，只选择障碍物进行膨胀
        obstacle_mask = (map_array == 1)
        # 使用圆形结构元进行膨胀
        struct = self._create_disk_structure(swell_radius_px)
        swelled_obstacle_mask = binary_dilation(obstacle_mask, structure=struct)

        # 在原地图基础上应用膨胀结果
        swelled_map = map_array.copy()
        swelled_map[swelled_obstacle_mask] = 1  # 将膨胀区域标记为障碍物

        return swelled_map

    def _create_disk_structure(self, radius: int):
        """创建一个用于二值形态学操作的圆形结构元。"""
        x, y = np.ogrid[-radius:radius+1, -radius:radius+1]
        mask = x**2 + y**2 <= radius**2
        return mask

    def _smooth_path(self, path: list) -> list:
        """
        使用B样条对路径进行平滑处理。
        B样条通常比标准三次样条能产生更平滑、曲率变化更连续的曲线。

        Args:
            path (list): 包含[x, y]坐标点的路径列表。

        Returns:
            list: 平滑后的路径点列表。
        """
        if len(path) < 4:
            return path  # 点太少，无法有效平滑

        try:
            path_arr = np.array(path)
            x_coords, y_coords = path_arr[:, 0], path_arr[:, 1]
            
            # 使用累积弧长作为参数t，这能更好地处理路径点分布不均的情况
            distances = np.sqrt(np.sum(np.diff(path_arr, axis=0)**2, axis=1))
            t = np.concatenate(([0], np.cumsum(distances)))
            t /= t[-1]  # 归一化到[0, 1]
            
            # 创建B样条插值器，k是样条阶数，最高为5阶以获得更好的平滑度
            k = min(5, len(path) - 1)
            spl_x = make_interp_spline(t, x_coords, k=k)
            spl_y = make_interp_spline(t, y_coords, k=k)
            
            # 在新的参数范围内生成更密集的点
            t_new = np.linspace(0, 1, len(path) * 4)
            x_smooth = spl_x(t_new)
            y_smooth = spl_y(t_new)
            
            return list(zip(x_smooth, y_smooth))
        except Exception:
            # 如果平滑失败（例如点共线），返回原始路径
            return path

    def generate_trajectory(self, swelled_map: np.ndarray, vehicle_state: dict, current_steer: float = 0) -> list:
        """
        从车辆当前状态生成一条轨迹。
        此方法是规划器的主要入口。它首先尝试高效的直线捷径，如果失败，则回退到
        更复杂的启发式搜索。

        Args:
            swelled_map (np.ndarray): 膨胀后的地图数组。
            vehicle_state (dict): 车辆当前状态，包含 'x', 'y', 'yaw'。

        Returns:
            list: 一系列[x, y]世界坐标点组成的轨迹。如果找不到路径，则返回空列表。
        """
        # # 1. 获取规划参数和车辆状态
        # target_path_len = self.plan_params['target_path_len']
        # lookahead_dist = self.plan_params['lookahead_dist']
        # current_x, current_y, current_yaw = vehicle_state['x'], vehicle_state['y'], vehicle_state['yaw']
        
        # # 2. 尝试直线捷径策略
        # start_col, start_row = self.map_obj._world_to_grid(current_x, current_y)
        # map_rows, map_cols = self.map_obj.resolution[1], self.map_obj.resolution[0]
        
        # # 2.1 检查起点是否有效
        # if 0 <= start_row < map_rows and 0 <= start_col < map_cols and swelled_map[start_row, start_col] != 1:
        #     straight_path = [[current_x, current_y]]
        #     num_steps = math.ceil(target_path_len / lookahead_dist) if lookahead_dist > 1e-6 else 0
        #     is_clear = True
            
        #     if num_steps > 0:
        #         temp_x, temp_y = current_x, current_y
        #         for _ in range(int(num_steps)):
        #             next_x = temp_x + lookahead_dist * math.cos(current_yaw)
        #             next_y = temp_y + lookahead_dist * math.sin(current_yaw)
                    
        #             start_c, start_r = self.map_obj._world_to_grid(temp_x, temp_y)
        #             next_c, next_r = self.map_obj._world_to_grid(next_x, next_y)
                    
        #             if _bresenham_line_check_numba(start_r, start_c, next_r, next_c, swelled_map):
        #                 is_clear = False
        #                 break
                    
        #             straight_path.append([next_x, next_y])
        #             temp_x, temp_y = next_x, next_y
            
        #     if is_clear and len(straight_path) > 1:
        #         return self._smooth_path(straight_path)

        # 3. 直线捷径失败，启动启发式搜索
        # 3.1 首先尝试前进
        path_nodes = self._find_path(swelled_map, vehicle_state, forward=True, prev_traj=self.previous_trajectory, start_steer=current_steer)
        
        # 3.2 如果前进失败（例如面朝墙壁），尝试后退
        if not path_nodes:
            path_nodes = self._find_path(swelled_map, vehicle_state, forward=False, prev_traj=self.previous_trajectory, start_steer=current_steer)
            if not path_nodes:
                self.previous_trajectory = None
                return []  # 前进和后退都无法找到路径

        # 4. 将路径节点转换为世界坐标
        world_path = [self._grid_to_world(node[0], node[1]) for node in path_nodes]
        
        # 5. 平滑并返回最终路径
        smooth_path = self._smooth_path(world_path)
        if smooth_path:
            self.previous_trajectory = smooth_path
        return smooth_path

    def _find_path(self, swelled_map: np.ndarray, vehicle_state: dict, forward: bool = True, prev_traj = None, start_steer = 0) -> list:
        """
        使用基于运动学模型的启发式搜索算法寻找路径。

        Args:
            swelled_map (np.ndarray): 膨胀后的地图。
            vehicle_state (dict): 车辆状态。
            forward (bool): True表示向前搜索，False表示向后搜索。

        Returns:
            list: 路径节点列表，每个节点为(col, row, yaw)。若失败则返回空列表。
        """

        if prev_traj is None:
            prev_traj = np.empty((0, 2), dtype=np.float64)
        else:
            prev_traj = np.array(prev_traj, dtype=np.float64)
        # 1. 预计算距离变换图，用于快速评估与障碍物的距离成本
        dist_transform = distance_transform_edt(swelled_map != 1)
        
        # 2. 获取规划参数
        lookahead = self.plan_params['lookahead_dist']
        target_len = self.plan_params['target_path_len']
        cost_w_steer = self.plan_params['cost_w_steer']
        cost_w_obs = self.plan_params['cost_w_obs']
        cost_w_reverse = self.plan_params['cost_w_reverse']
        cost_w_path_deviation = self.plan_params['cost_w_path_deviation']
        cost_w_steer_rate = self.plan_params['cost_w_steer_rate']
        cost_w_curvature = self.plan_params['cost_w_curvature']

        # 3. 初始化起始节点
        start_col, start_row = self.map_obj._world_to_grid(vehicle_state['x'], vehicle_state['y'])
        if swelled_map[start_row, start_col] == 1:
            return []  # 起点在障碍物内，无法规划
        
        start_yaw = (vehicle_state['yaw'] + np.pi) % (2 * np.pi) - np.pi
        start_node = (start_col, start_row, start_yaw, start_steer)
        
        # 4. 初始化搜索数据结构
        # 优先队列: (总成本, 路径长度, 节点元组)，成本越小优先级越高
        pq = [(0.0, 0.0, start_node)]
        
        # came_from字典用于重建路径，key=子节点, value=父节点
        came_from = {start_node: start_node}
        
        # 记录已访问的栅格(col, row)，避免在同一位置重复扩展，是一种剪枝策略
        visited_grid_cells = set()
        
        # 5. 封装地图参数以便传递给Numba函数
        map_params = {
            'scale_x': self.map_obj.scale_x, 'scale_y': self.map_obj.scale_y,
            'world_height': self.map_obj.world_size[1],
            'rows': self.map_obj.resolution[1], 'cols': self.map_obj.resolution[0]
        }

        # 6. 主搜索循环
        while pq:
            total_cost, path_len, current_node = heapq.heappop(pq)
            current_pos = (current_node[0], current_node[1])

            # 剪枝：如果当前栅格已处理过，则跳过
            if current_pos in visited_grid_cells:
                continue
            visited_grid_cells.add(current_pos)
            
            # 终止条件：达到目标路径长度
            if path_len >= target_len:
                return self._reconstruct_path(came_from, current_node, start_node)

            current_col, current_row, current_yaw, current_steer = current_node

            # 7. 顺序扩展节点
            for steer in self.steer_samples:
                is_valid, next_col, next_row, next_yaw, edge_cost = _process_steering_angle_numba(
                    current_col, current_row, current_yaw, steer, current_steer,
                    forward, lookahead,
                    map_params['scale_x'], map_params['scale_y'], map_params['world_height'],
                    map_params['rows'], map_params['cols'], swelled_map, dist_transform,
                    cost_w_steer, cost_w_obs, cost_w_reverse, self.wheelbase,
                    prev_traj, cost_w_path_deviation, cost_w_steer_rate, cost_w_curvature
                )
                if is_valid:
                    next_node = (next_col, next_row, next_yaw, steer)
                    new_cost = total_cost + edge_cost  # 增加lookahead作为路径长度成本，鼓励全面探索
                    new_len = path_len + lookahead
                    
                    # 使用next_node作为键，即使yaw是浮点数。
                    # 在实践中，由于计算确定性，这通常可行，但需注意浮点精度问题。
                    # 更鲁棒的方案是离散化yaw，但会增加复杂性。
                    if next_node not in came_from:
                        heapq.heappush(pq, (new_cost, new_len, next_node))
                        came_from[next_node] = current_node
        
        return []  # 搜索完成但未找到满足条件的路径
        
    def _reconstruct_path(self, came_from: dict, current_node, start_node) -> list:
        """
        从came_from字典中回溯，重建从起点到当前节点的路径。

        Args:
            came_from (dict): 存储父子关系的字典。
            current_node: 路径的终点。
            start_node: 路径的起点。

        Returns:
            list: 从起点到终点的路径节点列表。
        """
        path = []
        curr = current_node
        max_iters = len(came_from) + 1 # 安全检查，防止无限循环
        
        for _ in range(max_iters):
            path.append(curr)
            parent = came_from.get(curr)
            
            # 到达起点或遇到无效父节点时停止
            if parent is None or parent == curr:
                break
            
            curr = parent
        
        # 如果循环结束时不在起点，说明路径有问题
        if path[-1] != start_node:
            print(f"警告：路径重建未能回溯到起点。")
            return []

        path.reverse()
        return path

    def print_waypoints(self, map_array: np.ndarray, trajectory_world: list, vehicle_state: dict):
        """
        可视化地图、生成的轨迹和车辆当前位置。

        Args:
            map_array (np.ndarray): 要显示的地图（原始或膨胀后）。
            trajectory_world (list): [x, y] 航点列表。
            vehicle_state (dict): 当前车辆状态，用于绘制位置和朝向。
        """
        if map_array is None:
            print("错误：未提供用于可视化的地图数组。")
            return

        plt.figure(figsize=(12, 12))
        
        # 显示地图，origin='upper'使(0,0)在左上角，与数组索引匹配
        extent = [0, self.map_obj.world_size[0], 0, self.map_obj.world_size[1]]
        plt.imshow(map_array, cmap='gray_r', origin='upper', extent=extent)

        # 绘制生成的轨迹
        if trajectory_world:
            traj_x = [p[0] for p in trajectory_world]
            traj_y = [p[1] for p in trajectory_world]
            plt.plot(traj_x, traj_y, 'b-o', label='生成的轨迹', markersize=3, linewidth=1.5)

        # 绘制车辆位置和朝向
        vx, vy, vyaw = vehicle_state['x'], vehicle_state['y'], vehicle_state['yaw']
        plt.plot(vx, vy, 'ro', markersize=8, label='车辆位置')
        # 用箭头表示车辆朝向
        arrow_len = 5.0 # 箭头长度(米)
        plt.arrow(vx, vy,
                  arrow_len * math.cos(vyaw),
                  arrow_len * math.sin(vyaw),
                  head_width=2.0, head_length=2.5, fc='r', ec='r')

        plt.xlabel("世界坐标 X (米)")
        plt.ylabel("世界坐标 Y (米)")
        plt.title("轨迹生成可视化")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axis('equal')
        plt.show()

def benchmark_generate_trajectory(traj_gen, sw_map, vehicle_state, num_iters=100):
    """
    基准测试 generate_trajectory 的调用频率 (Hz)。
    """
    print(f"开始基准测试，迭代次数: {num_iters}...")
    start_t = time.time()
    for _ in range(num_iters):
        traj_gen.generate_trajectory(sw_map, vehicle_state)
    elapsed = time.time() - start_t
    hz = num_iters / elapsed if elapsed > 0 else float('inf')
    print(f"基准测试完成: {num_iters} 次调用耗时 {elapsed:.3f}s, 平均频率 {hz:.2f} Hz")


# ==============================================================================
# 示例用法和测试
# ==============================================================================

if __name__ == '__main__':
    import os
    from PIL import Image, ImageDraw

    # 确保Numba缓存目录存在且可写，如果环境受限可禁用缓存
    # import tempfile; numba.config.CACHE_DIR = tempfile.mkdtemp()
    # numba.config.DISABLE_CACHING = True # 用于调试

    map_file = "map.png"
    if not os.path.exists(map_file):
        print(f"测试地图 '{map_file}' 不存在，正在创建一个虚拟地图...")
        img = Image.new("RGB", (160, 160), "white")
        draw = ImageDraw.Draw(img)
        # 添加一个U形障碍物 (绿色在map_and_obstacle中被视为障碍)
        draw.rectangle([20, 20, 140, 40], fill=(0, 255, 0)) # 上边
        draw.rectangle([20, 20, 40, 120], fill=(0, 255, 0)) # 左边
        draw.rectangle([120, 20, 140, 120], fill=(0, 255, 0))# 右边
        img.save(map_file)

    # 1. 初始化地图处理器
    map_handler = map_and_obstacle(resolution_val=700, world_size_val=(160, 160)) # 2像素/米
    map_handler.load_map_from_png(map_file)
    static_map = map_handler.static_map_array

    # 2. 定义车辆和规划参数
    vehicle_params = {
        'width': 4.0,
        'max_steer': np.deg2rad(30), # 限制最大转向角，避免不切实际的转弯
        'wheelbase': 2.86
    }
    plan_params = {
        'target_path_len': 80,  # 目标路径长度(米)
        'lookahead_dist': 5.0,  # 步长(米)，较小的步长使路径更精细
        'num_steer_samples': 9, # 奇数个采样，包含0度
        'cost_w_steer': 5.0,    # 较高的转向惩罚，鼓励直行
        'cost_w_obs': 10.0,
        'cost_w_reverse': 100.0, # 非常高的倒车惩罚
        'cost_w_path_deviation': 20.0 # 路径偏离惩罚
    }
    
    # 定义车辆起始状态
    vehicle_start_state = {'x': 30, 'y': 50, 'yaw': -np.pi/2} # 从U形开口处，朝向U形内部

    # 3. 初始化轨迹生成器
    traj_gen = GenerateTrajectory(map_handler, vehicle_params, plan_params)

    # 4. 膨胀地图
    print("正在膨胀地图以考虑车辆尺寸...")
    swelled_map = traj_gen.get_and_swell_map(static_map)
    
    # 5. 生成轨迹
    print("正在生成轨迹...")
    start_time = time.time()
    trajectory = traj_gen.generate_trajectory(swelled_map, vehicle_start_state)
    print(f"轨迹生成耗时: {time.time() - start_time:.4f} 秒")

    # 6. 可视化结果
    if trajectory:
        print(f"成功生成一条包含 {len(trajectory)} 个点的轨迹。")
        traj_gen.print_waypoints(swelled_map, trajectory, vehicle_start_state)
    else:
        print("生成轨迹失败。")
        traj_gen.print_waypoints(swelled_map, [], vehicle_start_state)

    # --- 测试一个“被困”场景，强制车辆倒车 ---
    print("\n--- 测试“被困”场景 (需要倒车) ---")
    stuck_state = {'x': 80, 'y': 30, 'yaw': np.pi / 2} # 在U形底部，面朝墙壁
    
    print("正在为被困场景生成轨迹...")
    start_time = time.time()
    trajectory_stuck = traj_gen.generate_trajectory(swelled_map, stuck_state)
    print(f"轨迹生成耗时: {time.time() - start_time:.4f} 秒")

    if trajectory_stuck:
        print("成功从被困位置生成一条轨迹。")
        traj_gen.print_waypoints(swelled_map, trajectory_stuck, stuck_state)
    else:
        print("从被困位置生成轨迹失败。")
        traj_gen.print_waypoints(swelled_map, [], stuck_state)

    # --- 性能基准测试 ---
    print("\n--- 轨迹生成性能基准测试 ---")
    benchmark_generate_trajectory(traj_gen, swelled_map, vehicle_start_state, num_iters=50)