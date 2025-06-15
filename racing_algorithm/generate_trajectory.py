# Car: length: 5m, width: 2.2m, wheelbase: 2.86m, track: 1.69m, max_steering_angle: 35

import numpy as np
import heapq
import math
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation, distance_transform_edt
from scipy.interpolate import make_interp_spline
import time
import numba

try:
    from racing_algorithm.map_and_obstacle import map_and_obstacle
except ImportError:
    print("Error: Unable to import 'map_and_obstacle'. Please ensure map_and_obstacle.py is in the same directory.")
    exit()

# JIT-compile all the inner loops
@numba.njit(cache=True)
def _get_min_dist_to_path_numba(point_x, point_y, path_arr):
    if path_arr.shape[0] == 0:
        return 1e9
    min_d2 = 1e18
    for i in range(path_arr.shape[0]):
        dx = point_x - path_arr[i, 0]
        dy = point_y - path_arr[i, 1]
        d2 = dx*dx + dy*dy
        if d2 < min_d2:
            min_d2 = d2
    return math.sqrt(min_d2)

@numba.njit(cache=True)
def _get_position_from_left_right(current_x, current_y, yaw, swelled_map):
    left_yaw = yaw + np.pi/2
    rx = math.cos(left_yaw); ry = math.sin(left_yaw)
    right_yaw = yaw - np.pi/2
    r2x = math.cos(right_yaw); r2y = math.sin(right_yaw)

    rows, cols = swelled_map.shape
    threshold = max(rows, cols) // 5

    # left
    dl = threshold
    for i in range(1, threshold):
        cx = int(round(current_x + rx*2*i))
        cy = int(round(current_y - ry*2*i))
        if cx < 0 or cy < 0 or cx >= cols or cy >= rows or swelled_map[cy, cx]:
            dl = i
            break

    # right
    dr = threshold
    for i in range(1, threshold):
        cx = int(round(current_x + r2x*2*i))
        cy = int(round(current_y - r2y*2*i))
        if cx < 0 or cy < 0 or cx >= cols or cy >= rows or swelled_map[cy, cx]:
            dr = i
            break

    if dl + dr == 0:
        return 0.5
    return dr / (dl + dr)

@numba.njit(cache=True)
def _grid_to_world_numba(col, row, sx, sy, wh):
    return col/sx, wh - row/sy

@numba.njit(cache=True)
def _world_to_grid_numba(xw, yw, sx, sy, wh, rows, cols):
    c = xw * sx
    r = (wh - yw) * sy
    ci = int(round(c)); ri = int(round(r))
    if ci < 0: ci = 0
    elif ci >= cols: ci = cols-1
    if ri < 0: ri = 0
    elif ri >= rows: ri = rows-1
    return ci, ri

@numba.njit(cache=True)
def _bresenham_line_check_numba(r0, c0, r1, c1, swelled_map):
    r0 = int(r0); c0 = int(c0)
    r1 = int(r1); c1 = int(c1)
    dr = abs(r1 - r0); dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dc - dr
    rows, cols = swelled_map.shape
    r, c = r0, c0
    for _ in range(rows+cols):
        if r < 0 or c < 0 or r >= rows or c >= cols or swelled_map[r, c]:
            return True
        if r == r1 and c == c1:
            break
        e2 = 2*err
        if e2 >= -dr:
            err -= dr; c += sc
        if e2 <=  dc:
            err += dc; r += sr
    return False

@numba.njit(cache=True)
def _process_steering_angle_numba(
    c_col, c_row, c_yaw, steer, prev_steer, forward, lookahead,
    sx, sy, wh, rows, cols, sw_map, dist_tr,
    w_steer, w_obs, w_rev, wheelbase,
    prev_traj, w_dev, w_rate, w_curv,
    right_rate, left_rate_cost, cur_lr_rate
):
    dir_f = 1.0 if forward else -1.0
    xw, yw = _grid_to_world_numba(c_col, c_row, sx, sy, wh)
    dist = dir_f * lookahead

    if abs(steer) < 1e-6:
        nx = xw + dist * math.cos(c_yaw)
        ny = yw + dist * math.sin(c_yaw)
        nyaw = c_yaw
    else:
        R = wheelbase / math.tan(steer)
        beta = dist / R
        nyaw = c_yaw + beta
        # normalize
        nyaw = (nyaw + math.pi) % (2*math.pi) - math.pi
        cx = xw - R * math.sin(c_yaw)
        cy = yw + R * math.cos(c_yaw)
        nx = cx + R * math.sin(nyaw)
        ny = cy - R * math.cos(nyaw)

    ncol, nrow = _world_to_grid_numba(nx, ny, sx, sy, wh, rows, cols)
    if _bresenham_line_check_numba(c_row, c_col, nrow, ncol, sw_map):
        return False, 0,0,0.0,0.0,0.0

    cost_s  = w_steer * abs(steer)
    rate    = _get_position_from_left_right(ncol, nrow, nyaw, sw_map)
    new_lr  = 0.9*cur_lr_rate + 0.1*rate
    cost_lr = abs(new_lr - right_rate) * left_rate_cost
    curv    = abs(steer) / wheelbase
    cost_cv = w_curv * curv
    d2o     = dist_tr[nrow, ncol]
    cost_o  = w_obs / (d2o + 1e-6)
    cost_r  = 0.0 if forward else w_rev

    dev_cost = 0.0
    if prev_traj.shape[0] > 0:
        d2p = _get_min_dist_to_path_numba(nx, ny, prev_traj)
        dev_cost = w_dev * d2p

    total = cost_s + cost_o + cost_r + cost_cv + cost_lr + dev_cost + w_rate * abs((steer - prev_steer)/lookahead)
    return True, ncol, nrow, nyaw, total, new_lr

class GenerateTrajectory:
    def __init__(self, map_obj: map_and_obstacle, vehicle_params=None, plan_params=None):
        self.map_obj = map_obj
        # vehicle
        vp = vehicle_params or {}
        self.wheelbase = vp.get('wheelbase', 2.86)
        max_s = vp.get('max_steer', np.pi/6)
        # plan params flattened
        pp = plan_params or {}
        self.lookahead    = pp.get('lookahead_dist',    8.0)
        self.target_len   = pp.get('target_path_len',   30.0)
        self.w_steer      = pp.get('cost_w_steer',      5.0)
        self.w_obs        = pp.get('cost_w_obs',        10.0)
        self.w_rev        = pp.get('cost_w_reverse',    20.0)
        self.w_dev        = pp.get('cost_w_path_deviation', 20.0)
        self.w_rate       = pp.get('cost_w_steer_rate', 15.0)
        self.w_curv       = pp.get('cost_w_curvature',   10.0)
        self.right_rate   = pp.get('right_rate',        0.2)
        self.left_rate    = pp.get('left_cost_rate',    10.0)

        # steer samples, ensure odd count
        n = pp.get('num_steer_samples', 5)
        if n % 2 == 0: n+=1
        self.steer_samples = np.linspace(-max_s, max_s, n)

        # map constants
        self.sx, self.sy = map_obj.scale_x, map_obj.scale_y
        self.wh          = map_obj.world_size[1]
        self.rows_px, self.cols_px = map_obj.resolution[1], map_obj.resolution[0]

        # cache for distance transform
        self._last_map_id       = None
        self._cached_dist_tr    = None
        # cache for dilation struct
        self._disk_struct       = None

        self.previous_trajectory = None

    def get_and_swell_map(self, m: np.ndarray) -> np.ndarray:
        r_m = (self.map_obj.scale_x + self.map_obj.scale_y)/2.0
        radius = int(round((vehicle_params.get('width',4.0)/2.0) * r_m))
        if radius <= 0:
            return m.copy()
        if self._disk_struct is None or self._disk_struct.shape[0] != 2*radius+1:
            x,y = np.ogrid[-radius:radius+1, -radius:radius+1]
            self._disk_struct = (x*x + y*y) <= radius*radius

        obs = (m==1)
        d = binary_dilation(obs, structure=self._disk_struct)
        m2 = m.copy()
        m2[d] = 1
        return m2

    def generate_trajectory(self, sw_map: np.ndarray, vehicle_state: dict, current_steer: float=0.0) -> list:
        # cache distance transform
        mid = id(sw_map)
        if mid != self._last_map_id:
            self._last_map_id    = mid
            self._cached_dist_tr = distance_transform_edt(sw_map != 1)
        self.s_map = sw_map

        nodes = self._find_path(vehicle_state, current_steer)
        if not nodes:
            # try reverse
            nodes = self._find_path(vehicle_state, current_steer, forward=False)
            if not nodes:
                self.previous_trajectory = None
                return []

        # world path & smoothing
        world = [ _grid_to_world_numba(c,r,self.sx,self.sy,self.wh) for c,r,_,_,_ in nodes ]
        smooth = self._smooth_path(world)
        if smooth:
            self.previous_trajectory = smooth
        return smooth

    def _find_path(self, vehicle_state, start_steer, forward=True):
        prev = (np.empty((0,2),dtype=np.float64)
                if self.previous_trajectory is None
                else np.array(self.previous_trajectory,dtype=np.float64))
        dist_tr = self._cached_dist_tr

        sx, sy, wh = self.sx, self.sy, self.wh
        rows, cols = self.rows_px, self.cols_px
        lookahead = self.lookahead
        tlen      = self.target_len

        x, y, yaw = vehicle_state['x'], vehicle_state['y'], vehicle_state['yaw']
        c0, r0    = self.map_obj._world_to_grid(x, y)
        if self.s_map[r0, c0] == 1:
            return []

        # initial left/right rate
        lr0 = _get_position_from_left_right(c0, r0, (yaw+math.pi)%(2*math.pi)-math.pi, self.s_map)
        start = (c0, r0, (yaw+math.pi)%(2*math.pi)-math.pi, start_steer, lr0)

        pq = [(0.0, 0.0, start)]
        came = {start: start}
        visited = set()

        while pq:
            cost, plen, node = heapq.heappop(pq)
            ccol, crow, cyaw, csteer, clr = node
            if (ccol,crow) in visited: continue
            visited.add((ccol,crow))
            if plen >= tlen:
                return self._reconstruct_path(came, node, start)

            for st in self.steer_samples:
                ok, nc, nr, ny, ec, nlr = _process_steering_angle_numba(
                    ccol, crow, cyaw, st, csteer, forward, lookahead,
                    sx, sy, wh, rows, cols, self.s_map, dist_tr,
                    self.w_steer, self.w_obs, self.w_rev, self.wheelbase,
                    prev, self.w_dev, self.w_rate, self.w_curv,
                    self.right_rate, self.left_rate, clr
                )
                if not ok: 
                    continue
                new_cost = cost + ec
                new_len  = plen + lookahead
                nn = (nc, nr, ny, st, nlr)
                if nn not in came:
                    came[nn] = node
                    heapq.heappush(pq, (new_cost, new_len, nn))

        return []

    def _reconstruct_path(self, came, curr, start):
        path = []
        iters = len(came)+1
        for _ in range(iters):
            path.append(curr)
            parent = came.get(curr)
            if parent is None or parent == curr:
                break
            curr = parent
        path.reverse()
        if path and path[0] != start:
            return []
        return path

    def _smooth_path(self, path):
        if len(path) < 4:
            return path
        try:
            arr = np.array(path)
            xs, ys = arr[:,0], arr[:,1]
            dist = np.sqrt(np.sum(np.diff(arr,axis=0)**2,axis=1))
            t = np.concatenate(([0], np.cumsum(dist)))
            t /= t[-1]
            k = min(5, len(path)-1)
            spx = make_interp_spline(t, xs, k=k)
            spy = make_interp_spline(t, ys, k=k)
            tn = np.linspace(0,1,len(path)*4)
            return list(zip(spx(tn), spy(tn)))
        except Exception:
            return path

    def print_waypoints(self, map_array: np.ndarray, trajectory_world: list, vehicle_state: dict):
        if map_array is None:
            print("Fault: Invalid map array provided for visualization.")
            return

        plt.figure(figsize=(12, 12))
        
        extent = [0, self.map_obj.world_size[0], 0, self.map_obj.world_size[1]]
        plt.imshow(map_array, cmap='gray_r', origin='upper', extent=extent)

        if trajectory_world:
            traj_x = [p[0] for p in trajectory_world]
            traj_y = [p[1] for p in trajectory_world]
            plt.plot(traj_x, traj_y, 'b-o', label='Generated Trajectory', markersize=3, linewidth=1.5)

        vx, vy, vyaw = vehicle_state['x'], vehicle_state['y'], vehicle_state['yaw']
        plt.plot(vx, vy, 'ro', markersize=8, label='Vehicle Position')
        arrow_len = 5.0 # Arrow length (meters)
        plt.arrow(vx, vy,
                  arrow_len * math.cos(vyaw),
                  arrow_len * math.sin(vyaw),
                  head_width=2.0, head_length=2.5, fc='r', ec='r')

        plt.xlabel("World Coordinate X (m)")
        plt.ylabel("World Coordinate Y (m)")
        plt.title("Trajectory Generation Visualization")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.axis('equal')
        plt.show()

def benchmark_generate_trajectory(traj_gen, sw_map, vehicle_state, num_iters=100):

    print(f"Starting benchmark test with {num_iters} iterations...")
    start_t = time.time()
    for _ in range(num_iters):
        traj_gen.generate_trajectory(sw_map, vehicle_state)
    elapsed = time.time() - start_t
    hz = num_iters / elapsed if elapsed > 0 else float('inf')
    print(f"Benchmark test completed: {num_iters} calls took {elapsed:.3f}s, average frequency {hz:.2f} Hz")


if __name__ == '__main__':
    import os
    from PIL import Image, ImageDraw

    map_file = "map.png"
    if not os.path.exists(map_file):
        print(f"Test map '{map_file}' does not exist, creating a virtual map...")
        img = Image.new("RGB", (160, 160), "white")
        draw = ImageDraw.Draw(img)
        draw.rectangle([20, 20, 140, 40], fill=(0, 255, 0))
        draw.rectangle([20, 20, 40, 120], fill=(0, 255, 0))
        draw.rectangle([120, 20, 140, 120], fill=(0, 255, 0))
        img.save(map_file)

    map_handler = map_and_obstacle(resolution_val=700, world_size_val=(160, 160))
    map_handler.load_map_from_png(map_file)
    static_map = map_handler.static_map_array

    vehicle_params = {
        'width': 4.0,
        'max_steer': np.deg2rad(30),
        'wheelbase': 2.86
    }
    plan_params = {
        'target_path_len': 63,
        'lookahead_dist': 3.5,
        'num_steer_samples': 9,
        'cost_w_steer': 10,             # ↑ 保持车头方向
        'cost_w_obs': 0,                # ↑ 远离障碍物
        'cost_w_reverse': 100.0,        # ↑ 抑制倒车
        'cost_w_path_deviation': 0.07,  # ↑ 保持路径
        'cost_w_steer_rate': 8,         # ↑ 保持当前转向
        'cost_w_curvature': 8,          # ↑ 抑制大曲率
        'right_rate': 0.18,
        'left_cost_rate': 8
    }

    
    vehicles = [
        {'name': 'v1', 'x': 35, 'y': 30, 'yaw': 0, 'length': 10, 'width': 5},
        {'name': 'v2', 'x': 40, 'y': 120, 'yaw': math.pi/4, 'length': 12, 'width': 6},
        {'name': 'v3', 'x': 130, 'y': 30, 'yaw': -math.pi/2, 'length': 15, 'width': 4}
    ]
    map_handler.update_vehicle_obstacles(vehicles)
    
    vehicle_start_state = {'x': 30, 'y': 50, 'yaw': -np.pi/2}

    traj_gen = GenerateTrajectory(map_handler, vehicle_params, plan_params)

    swelled_map = traj_gen.get_and_swell_map(map_handler.map_and_obstacle_to_array())

    start_time = time.time()
    trajectory = traj_gen.generate_trajectory(swelled_map, vehicle_start_state)
    print(f"Trajectory takes: {time.time() - start_time:.4f} seconds")

    if trajectory:
        print(f"Successfully generated a trajectory with {len(trajectory)} points.")
        traj_gen.print_waypoints(swelled_map, trajectory, vehicle_start_state)
    else:
        print("Failed to generate trajectory.")
        traj_gen.print_waypoints(swelled_map, [], vehicle_start_state)

    print("\n---Test Stuck Scenario ---")
    stuck_state = {'x': 80, 'y': 30, 'yaw': np.pi / 2}

    print("Generating trajectory for stuck scenario...")
    start_time = time.time()
    trajectory_stuck = traj_gen.generate_trajectory(swelled_map, stuck_state)
    print(f"Trajectory takes: {time.time() - start_time:.4f} seconds")

    if trajectory_stuck:
        print("Successfully generated a trajectory from stuck position.")
        traj_gen.print_waypoints(swelled_map, trajectory_stuck, stuck_state)
    else:
        print("Failed to generate trajectory from stuck position.")
        traj_gen.print_waypoints(swelled_map, [], stuck_state)

    # --- Performance Benchmark ---
    print("\n---Trajectory Generation Performance Benchmark---")
    benchmark_generate_trajectory(traj_gen, swelled_map, vehicle_start_state, num_iters=50)