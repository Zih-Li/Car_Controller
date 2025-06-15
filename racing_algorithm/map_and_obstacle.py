import numpy as np
from PIL import Image, ImageDraw
import math
import matplotlib.pyplot as plt
import os
from skimage.draw import polygon as sk_polygon
Image.MAX_IMAGE_PIXELS = None
map_path = "map.png"

class map_and_obstacle:
    def __init__(self, resolution_val=1600, world_size_val=(160, 160), green_threshold_sq_val=30000):
        self.resolution = (resolution_val, resolution_val)
        
        self.scale_x = self.resolution[0] / self.world_size[0] if self.world_size[0] > 0 else 1
        self.scale_y = self.resolution[1] / self.world_size[1] if self.world_size[1] > 0 else 1

        self.obstacle_vehicles = []

        self.static_map_array = None
        self.map_image_path = None


        self.nontraversable_color_rgb = (0, 255, 0)
        self.green_color_distance_threshold_sq = green_threshold_sq_val

    def _world_to_grid(self, x_w, y_w):

        col_idx_float = x_w * self.scale_x
        

        row_idx_float = (self.world_size[1] - y_w) * self.scale_y
        
        col_idx = np.clip(int(round(col_idx_float)), 0, self.resolution[0] - 1)
        row_idx = np.clip(int(round(row_idx_float)), 0, self.resolution[1] - 1)
        
        return col_idx, row_idx


    def load_map_from_png(self, png_filepath):

        self.map_image_path = png_filepath
        try:
            raw_image = Image.open(png_filepath).convert("RGB")
            
            if raw_image.size != self.resolution:
                raw_image = raw_image.resize(self.resolution, Image.NEAREST)
            
            img_array = np.array(raw_image)
            
            target_color_rgb = np.array(self.nontraversable_color_rgb)
            squared_differences = (img_array.astype(np.int32) - target_color_rgb.astype(np.int32))**2
            sum_squared_differences = np.sum(squared_differences, axis=-1)
            nontraversable_pixels = sum_squared_differences < self.green_color_distance_threshold_sq

            yellow_rgb = np.array([255, 255, 0])
            yellow_diff = (img_array.astype(np.int32) - yellow_rgb.astype(np.int32))**2
            yellow_dist = np.sum(yellow_diff, axis=-1)
            yellow_pixels = yellow_dist < 2000

            self.static_map_array = np.zeros(self.resolution, dtype=np.float32)
            self.static_map_array[nontraversable_pixels] = 1
            self.static_map_array[yellow_pixels] = 1

            return True
        except FileNotFoundError:
            print(f"Error: Map PNG file not found at {png_filepath}")
            self.static_map_array = None
            return False
        except Exception as e:
            print(f"Error: Failed to load or process map PNG: {e}")
            self.static_map_array = None
            return False

    def update_vehicle_obstacles(self, vehicles_data_list):

        self.obstacle_vehicles = vehicles_data_list

    def add_obstacle(self):
        vehicle_obstacle_layer = np.zeros(self.resolution, dtype=np.uint8)

        for vehicle in self.obstacle_vehicles:
            center_x_w, center_y_w = vehicle['x'], vehicle['y']
            yaw_rad = vehicle['yaw']
            length_w, width_w = vehicle['length'], vehicle['width']

            hl, hw = length_w / 2.0, width_w / 2.0

            corners_vehicle_frame = [
                (hl, -hw),
                (hl, hw),
                (-hl, hw),
                (-hl, -hw) 
            ]

            rotated_corners_grid_coords = []
            for cvx, cvy in corners_vehicle_frame:
                x_rot_world_offset = cvx * math.cos(yaw_rad) - cvy * math.sin(yaw_rad)
                y_rot_world_offset = cvx * math.sin(yaw_rad) + cvy * math.cos(yaw_rad)
                
                x_world_corner = center_x_w + x_rot_world_offset
                y_world_corner = center_y_w + y_rot_world_offset
                
                g_col, g_row = self._world_to_grid(x_world_corner, y_world_corner)
                rotated_corners_grid_coords.append((g_row, g_col))

            if rotated_corners_grid_coords:
                row_coords = np.array([p[0] for p in rotated_corners_grid_coords])
                col_coords = np.array([p[1] for p in rotated_corners_grid_coords])
                
                rr, cc = sk_polygon(row_coords, col_coords, shape=vehicle_obstacle_layer.shape)
                
                vehicle_obstacle_layer[rr, cc] = 1
        
        return vehicle_obstacle_layer

    def map_and_obstacle_to_array(self):

        if self.static_map_array is None:
            if self.map_image_path:
                print("Error: Static map not loaded. Attempting to load from stored path...")
                if not self.load_map_from_png(self.map_image_path):
                    print("Error: Failed to load static map. Cannot generate combined map.")
                    return None
            else:
                print("Error: Static map not loaded. Please call load_map_from_png() first.")
                return None
        
        vehicle_obstacle_layer = self.add_obstacle()
        
        static_map = self.static_map_array.copy()

        combined_map = np.zeros_like(static_map, dtype=np.uint8)

        combined_map[(static_map == 1) | (vehicle_obstacle_layer == 1)] = 1

        return combined_map

    def print_map(self, map_array_to_print=None, title="Map Display"):

        import matplotlib.colors as mcolors

        if map_array_to_print is None:
            print("No map array provided to print_map.")
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
        bounds = [-0.1, 0.1, 1.1]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        plt.imshow(map_array_to_print, cmap=cmap, norm=norm, origin='upper', extent=extent_params)
        plt.xlabel(f"World X (left: 0, right: {self.world_size[0]})")
        plt.ylabel(f"World Y (bottom: 0, top: {self.world_size[1]})")
        plt.title(title)
        cbar = plt.colorbar(ticks=[0, 1])
        cbar.ax.set_yticklabels(['White (Passable)', 'Black (Barrier)'])
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
        plt.show()


if __name__ == "__main__":
    print("Running map_and_obstacle.py self-test...")

    try:
        test_img_size = (160, 160)
        img = Image.new("RGB", test_img_size, "white")
        draw = ImageDraw.Draw(img)
        
        draw.rectangle([20, 20, 60, 60], fill=(0,255,0))
        
        dummy_map_path = "dummy_map.png"

        img.save(dummy_map_path)
        print(f"Created dummy map at {dummy_map_path}")

        mapper = map_and_obstacle(resolution_val=1000, world_size_val=(160, 160))

        if os.path.exists(map_path):
            dummy_map_path = map_path
        
        if mapper.load_map_from_png(dummy_map_path):
            print("Static map loaded successfully.")

            vehicles = [
                {'name': 'v1', 'x': 0, 'y': 0, 'yaw': 0, 'length': 10, 'width': 5},
                {'name': 'v2', 'x': 40, 'y': 120, 'yaw': math.pi/4, 'length': 12, 'width': 6},
                {'name': 'v3', 'x': 130, 'y': 30, 'yaw': -math.pi/2, 'length': 15, 'width': 4}
            ]
            mapper.update_vehicle_obstacles(vehicles)
            print(f"Updated vehicle obstacles with {len(vehicles)} vehicles.")

            import time
            iterations = 100
            start_time = time.time()
            for _ in range(iterations):
                mapper.update_vehicle_obstacles(vehicles)
                _ = mapper.map_and_obstacle_to_array()
            elapsed = time.time() - start_time
            freq = iterations / elapsed if elapsed > 0 else float('inf')
            print(f"Generate test: {iterations} updates take {elapsed:.4f} s, frequency: {freq:.2f} Hz.")

            combined_map = mapper.map_and_obstacle_to_array()
            if combined_map is not None:
                print("Combined map with obstacles generated successfully.")
                mapper.print_map(combined_map, title=f"Combined Map ({mapper.resolution[0]}x{mapper.resolution[1]})")
            else:
                print("Failed to generate combined map.")
        else:
            print("Failed to load map.")

    except ImportError as e:
        print(f"ImportError: {e}. Please ensure Pillow, NumPy, Matplotlib, and scikit-image are installed.")
    except Exception as e:
        print(f"Error occurred in test script: {e}")
        import traceback
        traceback.print_exc()

