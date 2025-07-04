import json, os
import math
import argparse
import numpy as np
import time
import threading
from my_udp import UDPClient
from racing_algorithm.map_and_obstacle import map_and_obstacle
from racing_algorithm.generate_trajectory import GenerateTrajectory

# net信息从外部获取并手动，形如下字符串。
# 车长5m，车宽2.2m，轴距2.86m，轮距1.69m，最大转向角35

class Control:
    def __init__(self):
        net="7Acc2JOKeN4oNsgabHrJLodPeD6V,172.19.0.1,1052,1053"
        v_name, ip, udp, udp_s = net.split(",")
        self.udp_client = UDPClient(ip = ip, port = int(udp), send_port = int(udp_s), vehicle_name = v_name)

        self.m_v = 0
        self.m_x = 0
        self.m_y = 0
        self.m_yaw = 0
        self.traj = []
        self.L = 2.86 # length (meters) - adjust based on your vehicle
        self.min_lookahead = 3 # minimum lookahead distance
        self.max_lookahead = 9  # maximum lookahead distance
        self.lookahead_ratio = 0.4 # lookahead = speed * ratio + min_lookahead
        self.current_steer= 0
        self.control_rate = 100 # hz
        
        # self.trajectory = [] # This seems unused, self.traj is used
        self.json_file = "ref_route.json"
        self.map_to_load = "map.png"

        # Shared data and locks
        self.pose_lock = threading.Lock()
        self.map_data_lock = threading.Lock()
        self.trajectory_lock = threading.Lock()

        self.combined_map_data = None
        self.current_pose_for_planner = None # Stores {'x': ..., 'y': ..., 'yaw': ...}

        # Initialize mapper and trajectory generator
        self.mapper = map_and_obstacle(resolution_val=700, world_size_val=(160, 160))
        self.mapper.load_map_from_png(self.map_to_load)
        self.min_speed = 5.0
        self.max_speed = 12.0
        self.speed_rate = (self.max_speed - self.min_speed) / (math.pi / 6)   # Adjust speed based on curvature
        plan_params = {
            'target_path_len': 36,
            'lookahead_dist': 3,
            'num_steer_samples': 9,
            'cost_w_steer': 8,              # ↑ 保持车头方向
            'cost_w_obs': 2,                # ↑ 远离障碍物
            'cost_w_reverse': 100.0,        # ↑ 抑制倒车
            'cost_w_path_deviation': 0.1,   # ↑ 保持路径
            'cost_w_steer_rate': 8,         # ↑ 保持当前转向
            'cost_w_curvature': 2,          # ↑ 抑制大曲率
            'right_rate': 0.2,
            'left_cost_rate': 16,
            'rate_cost_dr': 0.01
        }
        self.traj_gen = GenerateTrajectory(
            map_obj=self.mapper,
            vehicle_params={"width": 6, 'max_steer': np.pi/5},
            plan_params=plan_params
        )

        self.last_steering_angle = 0
        self.steering_alpha = 0.95

    def _update_map_and_vehicle_state_loop(self):
        """
        Thread 1: Updates vehicle state (self.m_x, self.m_y, self.m_yaw)
                  and map data (self.combined_map_data, self.current_pose_for_planner).
        """
        start_time_loop = time.time()
        loop_count = 0
        print("Starting map and vehicle state update loop...")
        while True:
            iteration_start_time = time.time()
            
            vehicle_data = self.udp_client.get_vehicle_state()
            local_m_x = vehicle_data.x
            local_m_y = vehicle_data.y
            local_m_yaw = vehicle_data.yaw / 180 * math.pi

            with self.pose_lock:
                self.m_x = local_m_x
                self.m_y = local_m_y
                self.m_yaw = local_m_yaw
            
            vehicles = [{
                'name': obj.name,
                'x': obj.x, 'y': obj.y,
                'yaw': obj.yaw/180*math.pi,
                'length': 10, 'width': 5
            } for obj in self.udp_client.global_data]
            
            self.mapper.update_vehicle_obstacles(vehicles)
            combined = self.mapper.map_and_obstacle_to_array()
            current_planner_pose = {'x': local_m_x, 'y': local_m_y, 'yaw': local_m_yaw}

            # Removed map_data_lock for higher frequency
            self.combined_map_data = combined
            self.current_pose_for_planner = current_planner_pose

            loop_count += 1
            iteration_time = time.time() - iteration_start_time
            if iteration_time > 0:
                current_frequency = 1.0 / iteration_time
                # print(f"Map update frequency: {current_frequency:.2f} Hz")

            if loop_count % 100 == 0:
                total_time = time.time() - start_time_loop
                if total_time > 0:
                    average_frequency = loop_count / total_time
                    print(f"Average map update frequency over {loop_count} iterations: {average_frequency:.2f} Hz")
            
            # time.sleep(0.01) # Removed sleep for higher frequency

    def _trajectory_planning_loop(self):
        """
        Thread 2: Generates trajectory (self.traj) using map data.
        """
        start_time_loop = time.time()
        loop_count = 0
        print("Starting trajectory planning loop...")
        while True:
            iteration_start_time = time.time()
            
            current_map = None
            current_pos_for_traj = None

            # Removed map_data_lock for higher frequency
            # Reading potentially slightly older or concurrently updated data
            if self.combined_map_data is not None and self.current_pose_for_planner is not None:
                current_map = self.combined_map_data # shallow copy is fine for numpy array if not modified
                current_pos_for_traj = self.current_pose_for_planner.copy() # shallow copy for dict

            if current_map is not None and current_pos_for_traj is not None:
                new_traj = self.traj_gen.generate_trajectory(current_map, current_pos_for_traj, self.current_steer)
                # Removed trajectory_lock for higher frequency
                self.traj = new_traj
                
                loop_count += 1
                iteration_time = time.time() - iteration_start_time
                if iteration_time > 0:
                    current_frequency = 1.0 / iteration_time
                    # print(f"Trajectory generation frequency: {current_frequency:.2f} Hz")

                if loop_count % 5 == 0: # Print average frequency more often for this loop
                    total_time = time.time() - start_time_loop
                    start_time_loop = time.time() # Reset start time for frequency calculation
                    if total_time > 0:
                        average_frequency = loop_count / total_time
                        if average_frequency < 100:
                            print(f"Average trajectory generation frequency over {loop_count} iterations: {average_frequency:.2f} Hz")
                    loop_count = 0 # Reset loop count after printing average frequency
            else:
                # print("Waiting for map data for trajectory generation...")
                # time.sleep(0.05) # Removed sleep for higher frequency
                pass # Actively loop if data is not available, or add a very short sleep if CPU usage is too high


    def _vehicle_control_loop(self):
        """
        Thread 3: Main control loop. Reads vehicle state and trajectory, sends control commands.
        """
        print("Starting vehicle control loop...")
        loop_start_time = time.time()
        while True:
            control_iteration_start_time = time.time()

            local_x, local_y, local_yaw = 0, 0, 0
            with self.pose_lock:
                local_x = self.m_x
                local_y = self.m_y
                local_yaw = self.m_yaw
            
            current_trajectory_copy = []
            with self.trajectory_lock:
                if self.traj: # Ensure self.traj is not empty
                    current_trajectory_copy = list(self.traj) # Create a copy

            if not current_trajectory_copy:
                # print("Control loop: Trajectory is empty, sending zero commands.")
                v, turn_angle = 0, 0 # Or maintain last command, or specific behavior
            else:
                v, turn_angle = self.track_route(current_trajectory_copy, local_x, local_y, local_yaw)
            
            self.udp_client.send_control_command(v, turn_angle)
            self.current_steer = turn_angle

            elapsed = time.time() - control_iteration_start_time
            sleep_time = max(1.0/self.control_rate - elapsed, 0)
            time.sleep(sleep_time)
            # loop_start_time = time.time() # This was resetting the start time for frequency calculation, better to use fixed interval sleep

    def track_route(self, route_points, current_x, current_y, current_yaw):
        if not route_points:
            return 0, 0 
        v_point = (current_x, current_y)
        # Assuming self.m_v is the current actual speed of the vehicle.
        # Note: In the provided class structure, self.m_v is initialized to 0 and not updated 
        # from vehicle state. If self.m_v is always 0, lookahead_distance will be fixed to self.min_lookahead.
        current_speed_magnitude = abs(self.m_v) 
        lookahead_distance = min(max(self.min_lookahead + current_speed_magnitude * self.lookahead_ratio, self.min_lookahead), self.max_lookahead)

        # find_target_point will return the last point of the trajectory if no suitable forward point is found.
        target_point = self.find_target_point(route_points, v_point, current_yaw, lookahead_distance)
        
        if target_point is None: # Should be covered by the initial check if route_points was empty.
            return 0, 0

        dx = target_point[0] - v_point[0]
        dy = target_point[1] - v_point[1]
        
        l_d = math.sqrt(dx*dx + dy*dy)

        if l_d < 0.1: # Target is too close or coincident with current position.
                      # Stop the car to prevent erratic behavior or division by zero.
            return 0, 0 

        target_angle = math.atan2(dy, dx)
        
        # alpha_for_steering is the angle from the vehicle's current heading to the target point.
        alpha_for_steering = target_angle - current_yaw
        # Normalize alpha_for_steering to [-pi, pi]
        while alpha_for_steering > math.pi:
            alpha_for_steering -= 2 * math.pi
        while alpha_for_steering < -math.pi:
            alpha_for_steering += 2 * math.pi

        is_backing_target = False
        # If the absolute angle to the target is greater than ~95 degrees (pi/2 + 0.1 rad),
        # it implies the target point is generally behind the vehicle's current orientation.
        if abs(alpha_for_steering) > (math.pi / 2 + 0.1): 
            is_backing_target = True

        # Speed calculation
        # Curvature is based on the path's geometry relative to the car's forward axis.
        curvature = abs(2 * math.sin(alpha_for_steering) / l_d) # l_d is confirmed > 0.1
        
        if is_backing_target:
            # Define speed parameters for backing
            backing_min_speed = self.min_speed * 0.5 # Example: half of min forward speed
            backing_max_speed = self.min_speed       # Example: max backing speed is min forward speed
            # Adjust speed based on curvature, similar to forward motion but with backing limits.
            desired_speed_magnitude = max(backing_min_speed, backing_max_speed - self.speed_rate * curvature)
            v = -desired_speed_magnitude # Speed is negative for backing
        else:
            # Forward speed
            v = max(self.min_speed,  self.max_speed - self.speed_rate * curvature)

        # Steering calculation (Pure Pursuit)
        # The formula calculates wheel angle assuming forward motion.
        steering_angle_raw = math.atan2(2 * self.L * math.sin(alpha_for_steering), l_d)

        if is_backing_target:
            # When backing, to make the rear of the car turn towards the target direction
            # (indicated by alpha_for_steering), the front wheels need to steer in the opposite direction
            # of what they would for forward motion.
            steering_angle = -steering_angle_raw
        else:
            steering_angle = steering_angle_raw
            
        # Apply steering smoothing
        steering_angle = self.steering_alpha * steering_angle + (1 - self.steering_alpha) * self.last_steering_angle
        self.last_steering_angle = steering_angle
        
        turn_angle = steering_angle # Assuming turn_angle is the final steering command
        
        return v, turn_angle

    def find_target_point(self, route_points, vehicle_pos, vehicle_yaw, lookahead_distance):
        if not route_points:
            return None
        closest_idx = 0
        min_dist_sq = float('inf')
        for i, point in enumerate(route_points):
            dist_sq = (point[0] - vehicle_pos[0])**2 + (point[1] - vehicle_pos[1])**2
            if dist_sq < min_dist_sq:
                min_dist_sq = dist_sq
                closest_idx = i
        for i in range(closest_idx, len(route_points)):
            px, py = route_points[i]
            dx = px - vehicle_pos[0]
            dy = py - vehicle_pos[1]
            local_x = math.cos(-vehicle_yaw) * dx - math.sin(-vehicle_yaw) * dy
            if local_x < 0:
                continue
            dist = math.sqrt(dx*dx + dy*dy)
            if dist >= lookahead_distance:
                return (px, py)
        return route_points[-1]

    def start(self, args):
        """Starts all the threads."""
        self.udp_client.start()

        # Start the three main operational threads
        map_update_thread = threading.Thread(target=self._update_map_and_vehicle_state_loop, daemon=True)
        trajectory_planning_thread = threading.Thread(target=self._trajectory_planning_loop, daemon=True)      
        vehicle_control_thread = threading.Thread(target=self._vehicle_control_loop, daemon=True)
        if not args.parse_args().mode_1:
            trajectory_planning_thread.start()
        map_update_thread.start()
        vehicle_control_thread.start()

        print("All threads started.")

    def record_traj(self):
        traj = []
        while True:
            traj.append((self.m_x, self.m_y))
            time.sleep(0.1)
            with open("recorded_traj.json", "w") as f:
                json.dump(traj, f)

        
if __name__ == '__main__':
    args = argparse.ArgumentParser(description="Car Control Agent")
    args.add_argument('--mode_1', action='store_true', help="Run with static trajectory")
    args.add_argument('--record', action='store_true', help="Record trajectory")
    control = Control()
    if args.parse_args().mode_1:
        ref_route = json.load(open("ref_route.json", "r"))
        static_traj = []
        x = ref_route["X"]
        y = ref_route["Y"]
        for xi, yi in zip(x, y):
            static_traj.append((xi*10, yi*10))
            print(static_traj)
        control.traj = static_traj
    control.start(args) # Call the method to start all threads
    if args.parse_args().record:
        control.record_traj()
        print("Trajectory recording started.")
    # Keep the main thread alive to allow daemon threads to run
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Shutting down...")
