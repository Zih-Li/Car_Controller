import json, os
import math
import time
from my_udp import UDPClient


class Control:
    def __init__(self):
        
        net="7Acc2JOKeN4oNsgabHrJLodPeD6V,172.19.0.1,2326,2327"
        v_name, ip, udp, udp_s = net.split(",")
        self.udp_client = UDPClient(ip = ip, port = int(udp), send_port = int(udp_s), vehicle_name = v_name)

        self.m_v = 0
        self.m_x = 0
        self.m_y = 0
        self.m_yaw = 0

        # Pure Pursuit parameters
        self.L = 2.86 # length (meters) - adjust based on your vehicle
        self.min_lookahead = 6.0  # minimum lookahead distance
        self.max_lookahead = 12.0  # maximum lookahead distance
        self.lookahead_ratio = 0.8  # lookahead = speed * ratio + min_lookahead

        self.control_rate = 60  # hz

    def control_node(self, route):
        start_time = time.time()
        while True:
            vehicle_data = self.udp_client.get_vehicle_state()
            self.m_x = vehicle_data.x
            self.m_y = vehicle_data.y
            self.m_yaw = vehicle_data.yaw / 180 * math.pi

            v, w = self.track_route(route)
            self.udp_client.send_control_command(v, w)

            elapsed_time = time.time() - start_time
            sleep_time = max((1.0 / self.control_rate) - elapsed_time, 0.0)
            time.sleep(sleep_time)
            start_time = time.time()

    def track_route(self, route):
        with open(route) as route_file:
            data = json.load(route_file)
        
        route_points = [(x * 10, y * 10) for x, y in 
                        zip(data["X"], data["Y"])]
        
        v_point = (self.m_x, self.m_y)
        
        # Calculate dynamic lookahead distance
        current_speed = abs(self.m_v) if hasattr(self, 'm_v') else 10
        lookahead_distance = min(max(
            self.min_lookahead + current_speed * self.lookahead_ratio,
            self.min_lookahead), self.max_lookahead)
        
        # Find the target point using lookahead distance
        target_point = self.find_target_point(route_points, v_point, lookahead_distance)
        
        if target_point is None:
            print("No target found, stopping.")
            return 0, 0
        
        # Pure Pursuit calculation
        # Calculate alpha (angle between vehicle heading and target)
        dx = target_point[0] - v_point[0]
        dy = target_point[1] - v_point[1]
        target_angle = math.atan2(dy, dx)
        alpha = target_angle - self.m_yaw
        
        # Normalize alpha to [-pi, pi]
        while alpha > math.pi:
            alpha -= 2 * math.pi
        while alpha < -math.pi:
            alpha += 2 * math.pi
        
        # Calculate steering angle using Pure Pursuit formula
        # δ(t) = tan^(-1)(2L * sin(α(t)) / l_d)
        l_d = math.sqrt(dx*dx + dy*dy)  # actual distance to target
        
        if l_d < 0.1:  # avoid division by zero
            steering_angle = 0
        else:
            steering_angle = math.atan2(2 * self.L * math.sin(alpha), l_d)
        
        # Convert steering angle to angular velocity (approximate)
        # For a bicycle model: w = v * tan(δ) / L
        v = 15 # target speed
        w = v * math.tan(steering_angle) / self.L
        
        # Limit angular velocity to prevent sharp turns
        max_w = 2
        w = max(-max_w, min(max_w, w))
        
        return v, w

    def find_target_point(self, route_points, vehicle_pos, lookahead_distance):
        """Find the target point on the path using lookahead distance"""
        if not route_points:
            return None
        
        # Find the closest point on the path
        min_dist = float('inf')
        closest_idx = 0
        for i, point in enumerate(route_points):
            dist = math.sqrt((point[0] - vehicle_pos[0])**2 + (point[1] - vehicle_pos[1])**2)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i
        
        # Look for a point at lookahead distance starting from closest point
        for i in range(closest_idx, len(route_points)):
            point = route_points[i]
            dist = math.sqrt((point[0] - vehicle_pos[0])**2 + (point[1] - vehicle_pos[1])**2)
            
            if dist >= lookahead_distance:
                return point
        
        # If no point found at lookahead distance, return the last point
        return route_points[-1] if route_points else None


if __name__ == '__main__':
    control = Control()
    control.udp_client.start()
    control.control_node("./ref_route.json")