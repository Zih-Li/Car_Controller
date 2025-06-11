import json
import math
import time
from my_udp import UDPClient

# net信息从外部获取并手动，形如下字符串。
net="7Acc2JOKeN4oNsgabHrJLodPeD6V,172.19.0.1,4850,4851"

class Control:
    def __init__(self):
        
        net = net
        v_name, ip, udp, udp_s = net.split(",")
        self.udp_client = UDPClient(ip = ip, port = int(udp), send_port = int(udp_s), vehicle_name = v_name)

        self.m_v = 0
        self.m_x = 0
        self.m_y = 0
        self.m_yaw = 0

        self.control_rate = 10  # hz
        self.trajectory = []

    def control_node(self):
        start_time = time.time()
        json_file = "trajectory.json"
        
        while True:
            vehicle_data = self.udp_client.get_vehicle_state()
            self.m_x = vehicle_data.x
            self.m_y = vehicle_data.y
            self.m_yaw = vehicle_data.yaw / 180 * math.pi
            
            self.trajectory.append((self.m_x, self.m_y))
            
            with open(json_file, 'w') as f:
                json.dump(self.trajectory, f)
            
            v, w=10, 0
            self.udp_client.send_control_command(v, w)

            elapsed_time = time.time() - start_time
            sleep_time = max((1.0 / self.control_rate) - elapsed_time, 0.0)
            time.sleep(sleep_time)
            start_time = time.time()


if __name__ == '__main__':
    control = Control()
    control.udp_client.start()
    control.control_node()
