#!/usr/bin/env python
#-*-coding:utf-8-*-

""" [플래닝 테스트]
    
    1. select path 없을 때 감지 거리 늘리고 장애물 거리 순으로 path 고르기
    2. path_num 늘리기
    
    변경된 부분
    :   self.MARGINadd 추가
    :   generate_path에서 늘린 s 만큼은 따로 계산    def visual_selected(self, selected_path):
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
import os, sys

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
    
    def calc_consistency_cost(self, target_q, last_selected_q):
        consistency_cost = 0
        min_len = min(len(target_q), len(last_selected_q))
        diff = np.abs(target_q[:min_len] - last_selected_q[:min_len])
        
        consistency_cost = np.sum(diff) / len(last_selected_q) if len(last_selected_q) > 0 else 0
        return consistency_cost
    
    def __select_optimal_trajectory(self, candidate_paths):
        mincost = float('inf')
        select_path = None
        
        for fp in candidate_paths:
            if mincost >= fp.total_cost:
                mincost = fp.total_cost
                select_path = fp
           
        return select_path

    def optimal_trajectory(self, x, y, heading, qf=ROAD_WIDTH/2, path_num=3, path_len=3):
        if path_num == 3:
            self.S_MARGIN = 3  # 3차 사전주행 값 3
        else:
            self.S_MARGIN = 1.8 + 5   # 예선 : 1.8, 본선 : 5 # 3차 사전주행 값 5

        si, qi = self.glob_path.xy2sl(x, y)
        self.current_s = si
        self.current_q = qi
        ryaw = self.glob_path.get_current_reference_yaw_no_s()
        dtheta = heading - ryaw 
        
        safe_candidate_paths = self.generate_path(si, qi, dtheta, path_len, qf, path_num)
        selected_path = self.__select_optimal_trajectory(safe_candidate_paths)
       
        self.last_selected_path = selected_path
        
        ############### RVIZ 비쥬얼 코드 ##############
        if self.visual == True:
            self.visual_selected(selected_path)
        ##############################################

        return selected_path
