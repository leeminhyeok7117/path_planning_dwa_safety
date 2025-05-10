#!/usr/bin/env python
#-*-coding:utf-8-*-

""" [플래닝 테스트]
    
    1. select path 없을 때 감지 거리 늘리고 장애물 거리 순으로 path 고르기
    2. path_num 늘리기
    
    변경된 부분
    :   self.MARGINadd 추가
    :   generate_path에서 늘린 s 만큼은 따로 계산
    :   check_collision에서 가장 가까운 인덱스 찾기 추가
    
"""
    
# Python packages
import rclpy
from rclpy.qos import QoSProfile
import time
from sensor_msgs.msg import PointCloud
from geometry_msgs.msg import Point32
from std_msgs.msg import Float64
import numpy as np
from math import sin, cos, tan, pi, isnan
#from cv2 import getGaussianKernel
import os, sys
#sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from global_path import GlobalPath
import polynomial as polynomial
import frenet_path as frenet_path

# Cost Weight
W_OFFSET = 1 #safety cost 가중치
W_CONSISTENCY = 0.5 #smoothness cost 가중치
# MACARON_TREAD = 3 # 충돌 지름
ROAD_WIDTH = 3.0 # 예선 : 3.0 본선 : 4.0

#parameter
sl_d = 0.5      # sl 경로 사이의 거리 (m)

# mode 1은 곡률, mode 2 는 yaw값 비교
mode = 1

class TrajectoryPlanner: # path planner
    def __init__(self, node, gp_name):
        self.last_selected_path = frenet_path.Frenet_path() # for consistency cost
        self.glob_path = GlobalPath(gp_name)
        self.node = node
        
        qos_profile = QoSProfile(depth = 10)

        #중앙차선
        PATH_ROOT=(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))+"/path/npy_file/" #/home/gigi/catkin_ws/src/macaron_3/
        self.center = []
       
        self.candidate_pub = self.node.create_publisher(PointCloud,'/CDpath_tr', qos_profile)
        self.selected_pub = self.node.create_publisher(PointCloud,'/SLpath_tr', qos_profile)

        self.obstacle_time = 0
        self.visual = True

        self.current_s = 0
        self.current_q = 0

        self.S_MARGIN = 7 #5    # 생성한 경로 끝 추가로 경로 따라서 생성할 길이
        self.S_MARGINadd = 5
        self.collision_count = False


    def visual_candidate_5(self, candidate_paths):
        self.cd_path = PointCloud()

        for i in range(len(candidate_paths)):
            # print(candidate_paths[i].x[0])
            for j in range(len(candidate_paths[i].x)):
                p = Point32()
                p.x = candidate_paths[i].x[j]
                p.y = candidate_paths[i].y[j]
                p.z = 0.0
                self.cd_path.points.append(p)

        self.candidate_pub.publish(self.cd_path)

    def visual_selected(self, selected_path):
        self.sl_path = PointCloud()

        for i in range(len(selected_path.x)):
            p = Point32()
            p.x = selected_path.x[i]
            p.y = selected_path.y[i]
            p.z = 0.0
            self.sl_path.points.append(p)

        self.selected_pub.publish(self.sl_path)

    def generate_path(self, si, qi, dtheta, ds = 3, qf = ROAD_WIDTH/2, path_num = 3): 
        # (si, qi): 시작상태, dtheta: heading - ryaw, ds: polynomial의 길이, qf: 종료상태 q
        candidate_paths = [] # 후보경로들은 frenet_path class가 리스트 안에 담긴다. 
        sf_base = si + ds + self.S_MARGIN 
        sf_side = sf_base-1.0

        # generate path to each offset goal
        qf_arr =  np.linspace(qf, -qf, path_num) # 양수부터 차례대로 생성
        condition = [np.abs(qf_arr)<=0.1, (np.abs(qf_arr)>0.1) & (np.abs(qf_arr)<2.0), np.abs(qf_arr)>=2.0]
        choose = [sf_base+7.0, sf_base-1.0, sf_side]
        sf_arr = np.select(condition, choose)
        
        for qf_i, sf_i in zip(qf_arr, sf_arr):
            fp = frenet_path.Frenet_path()
            qs = polynomial.cubic_polynomial(si, qi, dtheta, ds, qf_i)  
            fp.s = np.arange(si, sf_i, sl_d)
            fp.q = qs.calc_point(fp.s) 
            
            fp.x, fp.y = self.glob_path.sl2xy(fp.s, fp.q)
            fp.yaw = self.glob_path.get_current_reference_yaw()
            fp.k = qs.calc_kappa(fp.s, self.glob_path.get_current_reference_kappa())

            # calculate path cost
            fp.offset_cost = abs(qf_i)
            fp.consistency_cost = self.calc_consistency_cost(fp.q, self.last_selected_path.q)
            fp.total_cost = W_CONSISTENCY * fp.consistency_cost + W_OFFSET * fp.offset_cost
            
            candidate_paths.append(fp)

        return candidate_paths
    
    # def generate_path_reverse(self, si, qi, dtheta, ds = 3, qf = ROAD_WIDTH/2, path_num = 3): 
    #     # (si, qi): 시작상태, dtheta: heading - ryaw, ds: polynomial의 길이, qf: 종료상태 q
    #     candidate_paths = [] # 후보경로들은 frenet_path class가 리스트 안에 담긴다.
    #     sf_final = si
    #     si = sf_final - ds - self.S_MARGIN

    #     # generate path to each offset goal
    #     for qf_ in np.linspace(qf, -qf, path_num): # 양수부터 차례대로 생성
    #         fp = frenet_path.Frenet_path() # 경로. 이 안에 모든 정보가 담긴다.
    #         qs = polynomial.cubic_polynomial(si, qi, dtheta, ds, qf_)  
    #         fp.s = [s for s in np.arange(si, sf_final, sl_d)]
    #         fp.q = [qs.calc_point(s) for s in fp.s]
    #         fp.q.reverse()
    #         print(fp.q)
    #         #######################################
    #         # 각 경로의 x, y, yaw, kappa계산
    #         for i in range(len(fp.s)): 
    #             x, y = self.glob_path.sl2xy(fp.s[i], fp.q[i])

    #             yaw = self.glob_path.get_current_reference_yaw()
    #             rkappa = self.glob_path.get_current_reference_kappa()
    #             fp.x.append(x)
    #             fp.y.append(y)
    #             path_yaw = yaw
    #             if path_yaw <= 0:
    #                 # print(path_yaw, 'pi')
    #                 path_yaw = 2 * pi + path_yaw
    #             fp.yaw.append(path_yaw)
    #             fp.k.append(qs.calc_kappa(fp.s[i], rkappa))
    #         #######################################
            
    #         fp.s.reverse()
    #         fp.k.reverse()
    #         fp.yaw.reverse()
    #         fp.x.reverse()
    #         fp.y.reverse()
            
    #         # calculate path cost
    #         fp.offset_cost = abs(qf_)
    #         fp.consistency_cost = self.calc_consistency_cost(fp.q, self.last_selected_path.q)
    #         fp.total_cost = W_CONSISTENCY * fp.consistency_cost + W_OFFSET * fp.offset_cost
            
    #         candidate_paths.append(fp)

    #     return candidate_paths
    
    def calc_consistency_cost(self, target_q, last_selected_q):
        consistency_cost = 0
        min_len = min(len(target_q), len(last_selected_q))
        diff = np.abs(target_q[:min_len] - last_selected_q[:min_len])
        
        consistency_cost = np.sum(diff) / len(last_selected_q) if len(last_selected_q) > 0 else 0
        return consistency_cost

    def __select_optimal_trajectory(self, candidate_paths, obs_xy,MACARON_TREAD):
        mincost = candidate_paths[0].total_cost
        select_path = None
        collision = False
        center_collision = False
        self.non_center = []
        num = 0

        for fp in candidate_paths:
            num += 1
            for xy in self.center:
                if self.check_center(xy[0], xy[1], fp.x, fp.y, MACARON_TREAD): #예선 : 4, 본선 : 6
                    center_collision = True
                    break

            if center_collision:
                self.non_center.append(num-1)
                #print(self.non_center)
                center_collision = False
                continue

            ################ 본선 
            for xy in obs_xy:
                check = self.check_collision(xy[0], xy[1], fp.x, fp.y, MACARON_TREAD)
                if check[0]:
                    collision = True
                    # print("충돌1"),num
                    break
            if collision :
                collision = False
                continue

            if mincost >= fp.total_cost:
                mincost = fp.total_cost
                select_path = fp
           
        return select_path

    def check_collision(self, obs_x, obs_y, target_xs, target_ys, MACARON_TREAD):
        dist = np.hypot(target_xs - obs_x, target_ys - obs_y)

        collision_detect = (dist <= (MACARON_TREAD/2))
        if np.any(collision_detect):
            print('장애물 감지!')
            self.current_q = 0
            return [True, np.argmax(collision_detect)]
        else:
            return [False, 999]

    def check_center(self, obs_x, obs_y, target_xs, target_ys,MACARON_TREAD):
        d = ((target_xs[4] - obs_x)**2 + (target_ys[4] - obs_y)**2)**0.5

        collision = (d <= (MACARON_TREAD/2))
        if collision:
                print('중앙선 침범!')
                return True

        return False

    def __select_longest_trajectory(self, candidate_paths, obs_xy, MACARON_TREAD):
        max_distance = 0
        original_candidate = candidate_paths
        #print(candidate_paths)
        candidate_paths = np.delete(candidate_paths,self.non_center)
        #print(candidate_paths)
        
        #try:
        select_path = candidate_paths[0]
        max_dis = 0
        for fp in candidate_paths:
            cur_distance = 0
            ##########################################
            fp.x.extend(fp.xplus)
            fp.y.extend(fp.yplus)
            ##########################################
            for xy in obs_xy:
                ##########################################
                collision, fp.obs_distance = self.check_collision(xy[0], xy[1], fp.x, fp.y, MACARON_TREAD, MODE=0)
                if not collision:
                    select_path = fp
                elif fp.obs_distance > max_dis:
                    select_path = fp

        return select_path

    def optimal_trajectory(self, x, y, heading, obs_xy, qf=ROAD_WIDTH/2, path_num=5, path_len=3, MACARON_TREAD=2, parking=0):
        # collision_count = False
        if path_num == 3:
            self.S_MARGIN = 3  # 3차 사전주행 값 3
        else:
            self.S_MARGIN = 1.8 + 5   # 예선 : 1.8, 본선 : 5 # 3차 사전주행 값 5

        si, qi = self.glob_path.xy2sl(x, y)
        self.current_s = si
        self.current_q = qi
        ryaw = self.glob_path.get_current_reference_yaw()
        dtheta = heading - ryaw

        if path_num == -1:
            safe_candidate_paths = self.generate_path_reverse(si, qi, dtheta, path_len, 0, 1)
        else:
            safe_candidate_paths = self.generate_path(si, qi, dtheta, path_len, 0, 1)
        
        if path_num == 1 or path_num == -1:
            if self.visual is True:
                self.visual_selected(safe_candidate_paths[0])
            return safe_candidate_paths[0]
            

        selected_path = self.__select_optimal_trajectory(safe_candidate_paths, obs_xy,MACARON_TREAD)
        if selected_path is None:
            # collision_count = True
            safe_candidate_paths = self.generate_path(si, qi, dtheta, path_len, qf, path_num)
            ############### RVIZ 비쥬얼 코드 ##############
            # if self.visual == True:
            #     self.visual_candidate_5(safe_candidate_paths)
            ##############################################
            selected_path = self.__select_optimal_trajectory(safe_candidate_paths, obs_xy,MACARON_TREAD)

            if selected_path is None:
                print("nothing is selected!!!!!!!!!!!!!!!!!")
                selected_path = self.__select_longest_trajectory(safe_candidate_paths,obs_xy,MACARON_TREAD)
        
        self.last_selected_path = selected_path
        ############### RVIZ 비쥬얼 코드 ##############
        if self.visual == True:
            self.visual_selected(selected_path)
            # self.max_curvature_pub(selected_path, collision_count, path_len, heading)
        ##############################################

        return selected_path
        # return selected_path.x, selected_path.y



    # def optimal_trajectory_parking(self, x, y, heading, obs_xy, qf = ROAD_WIDTH, path_num = 3, path_len = 5,MACARON_TREAD=1.5):
    #     self.collision_count = False
    #     self.S_MARGIN = 3
    #     si, qi = self.glob_path.xy2sl(x, y)
    #     self.current_s = si
    #     self.current_q = qi
    #     ryaw = self.glob_path.get_current_reference_yaw()
    #     dtheta = heading - ryaw
    #     safe_candidate_paths = self.generate_path(si, qi, dtheta, path_len, 0, 1)

    #     if path_num == 1:
    #         if self.visual == True:
    #             self.visual_selected(safe_candidate_paths[0])
    #             self.max_curvature_pub(safe_candidate_paths[0], self.collision_count, path_len, heading)
    #         return safe_candidate_paths[0], self.collision_count

    #     selected_path = self.__select_optimal_trajectory(safe_candidate_paths, obs_xy,MACARON_TREAD)
    #     if selected_path is None:
    #         self.collision_count = True
    #         print("collision 1")
    #         safe_candidate_paths = self.generate_path(si, qi, dtheta, path_len, qf, path_num)
    #         ############### RVIZ 비쥬얼 코드 ##############
    #         if self.visual == True:
    #             self.visual_candidate_5(safe_candidate_paths)
    #         ##############################################
    #         selected_path = self.__select_optimal_trajectory(safe_candidate_paths, obs_xy,MACARON_TREAD)

    #         if selected_path is None:
    #             print("collision 2")
    #             self.collision_count = True
    #             selected_path = self.__select_longest_trajectory(safe_candidate_paths,obs_xy,MACARON_TREAD)
        
    #     self.last_selected_path = selected_path
    #     ############### RVIZ 비쥬얼 코드 ##############
    #     if self.visual == True:
    #         self.visual_selected(selected_path)
    #         self.max_curvature_pub(selected_path, self.collision_count, path_len, heading)

    #     ##############################################

    #     selected_path = safe_candidate_paths[0]

    #     return selected_path, self.collision_count
