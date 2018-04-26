# -*- coding: utf-8 -*-

import numpy as np
import scipy.linalg


class KalmanFilter(object):

    def __init__(self, measurement):
        _n_state = 8 #状态数,分别为(x,y,a,h,dx,dy,da,dh)
        _n_measure = 4 #观测数

        '''
        A: transitionMatrix
        H: measurementMatrix
        Q: processNoiseCov
        R: measurementNoiseCov
        '''
        self.A = np.eye(_n_state, _n_state)
        for i in range(_n_state - _n_measure):
            self.A[i, _n_measure + i] = 1.
        self.H = np.eye(_n_measure, _n_state)
            
        # 依据当前状态动态的决定Q矩阵和R矩阵
        self._std_weight_position = 1. / 20
        self.Q = None
        self._std_weight_velocity = 1. / 160
        self.R = None

        # 初始化state和covariance
        self.state = np.r_[measurement, np.zeros_like(measurement)]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]]
        self.covariance = np.diag(np.square(std))

    def predict(self):
        # 执行time update
        std = [
            self._std_weight_position * self.state[3],
            self._std_weight_position * self.state[3],
            1e-2,
            self._std_weight_position * self.state[3],
            self._std_weight_velocity * self.state[3],
            self._std_weight_velocity * self.state[3],
            1e-5,
            self._std_weight_velocity * self.state[3]]
        self.Q = np.diag(np.square(std))

        self.state = np.dot(self.A, self.state)
        self.covariance = np.linalg.multi_dot((self.A, self.covariance, self.A.T)) + self.Q

    def project(self):
        # state space 映射到 measurement space

        std = [
            self._std_weight_position * self.state[3],
            self._std_weight_position * self.state[3],
            1e-1,
            self._std_weight_position * self.state[3]]
        self.R = np.diag(np.square(std))

        state = np.dot(self.H, self.state)
        covariance = np.linalg.multi_dot((self.H, self.covariance, self.H.T)) + self.R

        return state, covariance

    def correct(self, measurement):
        #import pdb
        #pdb.set_trace()
        # 执行measurement update
        projected_state, projected_covariance = self.project()

        chol_factor, lower = scipy.linalg.cho_factor(projected_covariance, lower=True, check_finite=False)
        KG = scipy.linalg.cho_solve(
            (chol_factor, lower), np.dot(self.covariance, self.H.T).T,
            check_finite=False).T

        self.state = self.state + np.dot(measurement - projected_state, KG.T)
        self.covariance = self.covariance - np.linalg.multi_dot((KG, projected_covariance, KG.T))


    def gating_distance(self, measurements):
        # 计算 state 与 measurement 之间的 Mahalanobis distance, 并在使用9.4877作为阈值进行过滤
        state, covariance = self.project()

        cholesky_factor = np.linalg.cholesky(covariance)
        d = measurements - state
        z = scipy.linalg.solve_triangular(
            cholesky_factor, d.T, lower=True, check_finite=False,
            overwrite_b=True)
        squared_maha = np.sum(z * z, axis=0)
        return squared_maha
