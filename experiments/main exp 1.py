import json
import math
import time
from my_udp import UDPClient
from pynput import keyboard

class Control:
    def __init__(self):
        net="7Acc2JOKeN4oNsgabHrJLodPeD6V,172.19.0.1,2708,2709"
        v_name, ip, udp, udp_s = net.split(",")
        self.udp_client = UDPClient(ip = ip, port = int(udp), send_port = int(udp_s), vehicle_name = v_name)

        # 控制状态
        self.forward = False
        self.backward = False
        self.left = False
        self.right = False
        self.brake = False
        
        # 速度参数
        self.max_speed = 20.0
        self.normal_speed = 10.0
        self.turn_rate = 1.0
        self.control_rate = 10  # hz
        
        # 设置键盘监听
        self.listener = keyboard.Listener(
            on_press=self.on_press,
            on_release=self.on_release)
        self.listener.daemon = True
        self.listener.start()

    def on_press(self, key):
        try:
            if key.char.lower() == 'w':
                self.forward = True
                self.backward = False  # 防止前后同时按下
            elif key.char.lower() == 's':
                self.backward = True
                self.forward = False  # 防止前后同时按下
            elif key.char.lower() == 'a':
                self.left = True
            elif key.char.lower() == 'd':
                self.right = True
        except AttributeError:
            if key == keyboard.Key.space:
                self.brake = True
                self.forward = False
                self.backward = False
            elif key == keyboard.Key.esc:
                return False

    def on_release(self, key):
        try:
            if key.char.lower() == 'w':
                self.forward = False
            elif key.char.lower() == 's':
                self.backward = False
            elif key.char.lower() == 'a':
                self.left = False
            elif key.char.lower() == 'd':
                self.right = False
        except AttributeError:
            if key == keyboard.Key.space:
                self.brake = False

    def control_node(self):
        start_time = time.time()
        
        while True:
            # 计算线速度
            v = 0.0
            if self.forward:
                v = self.normal_speed
            elif self.backward:
                v = -self.normal_speed
                
            # 如果刹车被按下，速度为0
            if self.brake:
                v = 0.0
                
            # 计算角速度
            w = 0.0
            if self.left:
                w = self.turn_rate
            if self.right:
                w = -self.turn_rate
            self.udp_client.send_control_command(v, w)

            elapsed_time = time.time() - start_time
            sleep_time = max((1.0 / self.control_rate) - elapsed_time, 0.0)
            time.sleep(sleep_time)
            start_time = time.time()

if __name__ == '__main__':
    control = Control()
    control.udp_client.start()
    control.control_node()