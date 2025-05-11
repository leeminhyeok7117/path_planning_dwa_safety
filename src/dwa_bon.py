#!/usr/bin/env python3
# -*-coding:utf-8-*-

from math import *

import numpy as np
import rclpy
from rclpy.qos import QoSProfile
from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud
from std_msgs.msg import Float64

from global_path import GlobalPath

"""
mode 0: 차선 변경 없이 한 차선 안에서 좁은 범위의 정적 장애물 회피
mode 1: 차선을 변경 해야 하는 넓은 범위의 정적 장애물 회피 (왼쪽 차선 -> 오른쪽 차선)
mode 2: gps mapping 을 이용한 track 에서 rubber cone 회피
mode 3: 유턴
"""

"""
class DWA_visual(Node):
    def __init__(self):
        super().__init__('dwa')"""
        
class DWA:
    def __init__(self, node, gp_name):
        
        qos_profile = QoSProfile(depth=10)
        self.node = node
        self.candidate_pub = node.create_publisher(PointCloud,'/CDpath2', qos_profile)
        self.selected_pub = node.create_publisher(PointCloud,'/SLpath2',  qos_profile)
        
        self.glob_path = GlobalPath(gp_name)

        self.visual = True
        self.cd_path = None
        self.sel_path = None

        # 로봇의 운동학적 모델 상수 설정
        self.max_speed = 8.0  # 최고 속도 [m/s]
        self.max_steer = np.deg2rad(27.0)  # 27도 [deg]
        self.max_a = 1.0  # 내가 정하면 됨 [m/s^2]
        self.max_steer_a = np.deg2rad(25.0)  # 내가 정하면 됨 [deg/s^2]

        self.length = 2.02  # 차 길이 [m]
        self.width = 1.16  # 차 폭 [m]
        self.tread = 0.985  # 같은 축의 바퀴 중심 간 거리 [m]
        self.wheel_radius = 0.165  # 바퀴 반지름 [m]
        self.wheel_base = 1.04  # 차축 간 거리 [m]

        self.cur_steer = 0
        self.current_s = 0
        self.current_q = 0

        self.predict_time = 0.7  # 미래의 위치를 위한 예측 시간
        self.search_frame = 5  # 정수로 입력 (range 에 사용)
        self.DWA_search_size = [3, 21]  # Dynamic Window 에서 steer 의 분할 수 (홀수 and 정수)
        self.obstacle_force = 2.0  # 2m

        self.flag = 0
        self.num = 0

        self.max_index = len(self.glob_path.rx)

        self.obstacle_force_uturn = 3.0
        
        self.detected_points=[]
        self.store_line = []

        # ###################### 가중치 #########################
        # mode == 1 
        self.mode_1_w1 = 10  # global path 와의 이격
        self.mode_1_w2 = 10  # obs 와의 거리
        self.mode_1_w3 = 3

        self.mode_2_w1 = 1
        self.mode_2_w2 = 10

        self.mode_3_w1 = 10
        self.mode_3_w2 = 10

        self.max_cost = 100

        # ######################################################

    # ↓↓ 비주얼 코드 ↓↓
    def visual_candidate_paths(self, candidate_paths):
        self.cd_path = PointCloud()
        
        # 헤더 id 필요하려나?
        for i in range(len(candidate_paths)):
            for j in range(len(candidate_paths[i])):
                p = Point32()
                p.x = candidate_paths[i][j][0]
                p.y = candidate_paths[i][j][1]
                p.z = 0.0
                self.cd_path.points.append(p)
        self.candidate_pub.publish(self.cd_path)

    def visual_selected_path(self, selected_path):
        self.sel_path = PointCloud()
        for i in range(len(selected_path)):
            p = Point32()
            p.x = selected_path[i][0]
            p.y = selected_path[i][1]
            p.z = selected_path[i][2]
            self.sel_path.points.append(p)
        self.selected_pub.publish(self.sel_path)

    # ↑↑ 비주얼 코드 ↑↑

    # noinspection PyMethodMayBeStatic
    def convert_coordinate_l2g(self, d_x, d_y, d_theta):  # local -> global 좌표 변환 함수
        d_theta = -pi / 2 + d_theta
        x_global = cos(d_theta) * d_x - sin(d_theta) * d_y
        y_global = sin(d_theta) * d_x + cos(d_theta) * d_y
        theta_global = pi /2 + d_theta
        return np.stack([x_global, y_global, theta_global], axis=1)
        # return : local coordinate 에서의 [d_x, d_y, d_theta] 를 global coordinate 에서의 [d_x', d_y', d_theta'] 로 반환

    def generate_predict_point(self, x, y, velocity, steer, heading, mode):  # local 좌표계 에서 예측 점 좌표를 구해서 global 좌표로 return
            # 접선 이동 거리 (= 호의 길이로 사용할 예정)
        tan_dis = min((velocity * self.predict_time / 3.6 + 0.1, 0.9, 0.8)[mode], 1.0)
            # Assuming Bicycle model, (곡률 반경) = (차축 간 거리) / tan(조향각)
            
        R = self.wheel_base / tan(-steer) if steer != 0.0 else float('inf')
        theta_arr, future_pos = [], []
        frame_arr = np.arange(self.search_frame)
        if np.isinf(R):
            theta_arr = np.zeros_like(frame_arr, dtype=float)
            dx = np.zeros_like(frame_arr, dtype=float)
            dy = tan_dis * (frame_arr + 1)
        else:
            theta_arr = np.cumsum(np.full(frame_arr, tan_dis / R))
            dx = R * (1 - np.cos(theta_arr))
            dy = R * np.sin(theta_arr)

        future_pos = self.convert_coordinate_l2g(dx, dy, theta_arr + heading)
        future_pos[:, 0] += x
        future_pos[:, 1] += y
        return future_pos  # return 값은 global coordinate 의 예측 점 x, y 좌표  -> [[x1, y1, theta1], [x2, y2, theta2], .....]

    def calc_dynamic_window(self, velocity): #steer=0.0고정
        DWA_steer = np.linspace(-self.max_steer, self.max_steer, self.DWA_search_size[1])
        DWA_velocity = velocity + self.max_a
        dw = [DWA_velocity, DWA_steer]
        return dw
    
    def cost_function(self, pose, obs_xy, mode):
        cost1, cost2 = 0.0, 0.0
        obs_xy = np.zeros((1,2)) if len(obs_xy) else np.asarray(obs_xy)

        if mode == 0:
            gp_separation = self.glob_path.xy2sl(pose[-1, 0], pose[-1, 1])[1]
            cost1 = abs(gp_separation/2.3) if -0.5 <= gp_separation <= 0.5 else abs(gp_separation)
            return cost1
        
        elif mode == 1: # mini_obs mode
            gp_separation = self.glob_path.xy2sl(pose[-1, 0], pose[-1, 1])[1]
            cost1 = abs(gp_separation/2.3) if -1.5 <= gp_separation <= 2.0 else abs(gp_separation * 30)
            
            points = pose[:, :2]
            dx = points[:, 0:1] - obs_xy[np.newaxis, :, 0]
            dy = points[:, 1:2] - obs_xy[np.newaxis, :, 1]
            try:   
                obs_d = np.min(np.hypot(dx, dy),axis=1)[1:]
            except ValueError:
                obs_d = self.obstacle_force
                
            if np.any((obs_d > 0.0) & (obs_d <= 0.5)):
                cost2 = self.max_cost
            else: 
                possible = obs_d < self.obstacle_force
                cost2 = np.sum((self.obstacle_force - obs_d[possible]) / self.obstacle_force)
            return cost1*self.mode_1_w1 + cost2*self.mode_1_w2
            
        elif mode == 2: # bis_obs mode
            gp_separation = self.glob_path.xy2sl(pose[-1, 0], pose[-1, 1])[1]
            cost1 = abs(gp_separation/2.3) if -3.5 <= gp_separation <= 0.1 else abs(gp_separation * 100)
            
            obs_xy = np.asarray(obs_xy)
            points = pose[:, :2]
            dx = points[:, 0:1] - obs_xy[np.newaxis, :, 0]
            dy = points[:, 1:2] - obs_xy[np.newaxis, :, 1]
            try:   
                obs_d = np.min(np.hypot(dx, dy),axis=1)[1:]
            except ValueError:
                obs_d = self.obstacle_force
                
            if np.any((obs_d > 0.0) & (obs_d <= 1.0)):
                cost2 = self.max_cost
            else: 
                possible = obs_d < self.obstacle_force
                cost2 = np.sum((self.obstacle_force - obs_d[possible]) / self.obstacle_force)
            return cost1*self.mode_1_w1 + cost2*self.mode_1_w2
            
    def cost_function_global(self, pose, target_index):
        target_x, target_y = self.glob_path.rx[target_index], self.glob_path.ry[target_index]
        cost = np.hypot(target_x - pose[-1, 0], target_y - pose[-1, 1])
        return cost
            
    def DWA(self, x, y, heading, obs_xy=None, mode=0, current_s=0, cur_speed=0, dwa_mode=0):  # (차량의 x, y, heading), (장애물의 x, y)
        best_cost = float('inf')
        candidate_paths, selected_path = [], []
    
        # crusing mode
        if dwa_mode == 0:

            self.current_pose = [x, y, heading]            

            dw = self.calc_dynamic_window(cur_speed)
            velocity = dw[0]

            self.glob_path.cur_ref_index = current_s
            path_dis = (cur_speed * self.predict_time / 3.6 + 0.1) * 5
            if path_dis > 5: path_dis = 5
            index_offset = path_dis * 10
            target_index = int(current_s + index_offset)

            for steer in dw[1]:
                future_pos = self.generate_predict_point(x, y, velocity, steer, heading, dwa_mode)
                
                candidate_paths.append(future_pos)
                try:
                    cost = self.cost_function_global(future_pos, target_index)
                except:
                    cost = self.cost_function_global(future_pos, -1)


                if cost < best_cost:
                    best_cost = cost
                    selected_path = future_pos

        # mini_obs mode
        elif dwa_mode == 1:
            dw = self.calc_dynamic_window(cur_speed)
            velocity = dw[0]
            for steer in dw[1]:
                future_pos = self.generate_predict_point(x, y, velocity, steer, heading, dwa_mode)
                
                candidate_paths.append(future_pos)
                cost = self.cost_function(future_pos, obs_xy=obs_xy, mode=mode)

                if cost < best_cost:
                    best_cost = cost
                    selected_path = future_pos

        # big_obs mode
        elif dwa_mode == 2:
            dw = self.calc_dynamic_window(cur_speed)
            velocity = dw[0]
            i = 0
            for steer in dw[1]:
                future_pos = self.generate_predict_point(x, y, velocity, steer, heading, dwa_mode)
                
                candidate_paths.append(future_pos)
                cost = self.cost_function(future_pos, obs_xy=obs_xy, mode=mode)

                if cost < best_cost:
                    best_cost = cost
                    selected_path = future_pos
                i+=1

        if self.visual:
            self.visual_candidate_paths(candidate_paths)
            self.visual_selected_path(selected_path)

        return selected_path
    

