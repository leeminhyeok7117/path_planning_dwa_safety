#!/usr/bin/env python
#-*-coding:utf-8-*-

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

class TrajectoryPlanner: # path planner
    def __init__(self, node, gp_name):
        self.last_selected_path = frenet_path.Frenet_path() # for consistency cost
        self.glob_path = GlobalPath(gp_name)
        self.node = node
        
        qos_profile = QoSProfile(depth = 10)
       
        self.selected_pub = self.node.create_publisher(PointCloud,'/SLpath_tr', qos_profile)

        self.visual = True

        self.current_s = 0
        self.current_q = 0

        self.S_MARGIN = 7 #5    # 생성한 경로 끝 추가로 경로 따라서 생성할 길이

    def visual_selected(self, selected_path):
        self.sl_path = PointCloud()

        for i in range(len(selected_path.x)):
            p = Point32()
            p.x = selected_path.x[i]
            p.y = selected_path.y[i]
            p.z = 0.0
            self.sl_path.points.append(p)

        self.selected_pub.publish(self.sl_path)

    def generate_center_path(self, si, qi, dtheta, ds = 3): 
        # (si, qi): 시작상태, dtheta: heading - ryaw, ds: polynomial의 길이, qf: 종료상태 q
        sf_base = si + ds + self.S_MARGIN 
        sl_d = 0.5

        qf_i = 0.0
        sf_i = sf_base+7.0

        fp = frenet_path.Frenet_path()
        qs = polynomial.cubic_polynomial(si, qi, dtheta, ds, qf_i)  
        fp.s = np.arange(si, sf_i, sl_d)
        fp.q = qs.calc_point(fp.s) 
        
        fp.x, fp.y = self.glob_path.sl2xy(fp.s, fp.q)
        fp.yaw = self.glob_path.get_current_reference_yaw()
        fp.k = qs.calc_kappa(fp.s, self.glob_path.get_current_reference_kappa())

        return fp

    def optimal_trajectory(self, x, y, heading, path_len=3):

        self.S_MARGIN = 1.8 + 5   # 예선 : 1.8, 본선 : 5 # 3차 사전주행 값 5

        si, qi = self.glob_path.xy2sl(x, y)
        self.current_s = si
        self.current_q = qi
        ryaw = self.glob_path.get_current_reference_yaw_no_s()
        dtheta = heading - ryaw 
        
        selected_path = self.generate_center_path(si, qi, dtheta, path_len)
        
        ############### RVIZ 비쥬얼 코드 ##############
        if self.visual == True:
            self.visual_selected(selected_path)
        ##############################################

        return selected_path