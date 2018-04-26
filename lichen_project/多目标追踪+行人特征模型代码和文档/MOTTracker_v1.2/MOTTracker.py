# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import argparse
import os

import cv2
import numpy as np
from sklearn.utils.linear_assignment_ import linear_assignment

import kalman_filter


class Detection(object):
    def __init__(self, tlwh, confidence, feature):
        self.tlwh = tlwh # bounding box position, (top left x, top left y, width, height)
        self.confidence = confidence # detection confidence
        self.feature = feature # detection feature

    def to_xyah(self):
        # 将 bounding box 转化为 (center x, center y, ratio, height) 的格式，以使用 kalman_filter
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


class Tracker(object):
    # 单目标追踪类
    def __init__(self, measurement, track_id, feature=None):
        self.track_id = track_id # track id
        self.time_since_update = 0 # 距上次匹配成功的时长
        self.features = [] # 收集到的feature
        if feature is not None:
            self.features.append(feature)
        '''
        state: 目标状态
          0: 初始化阶段，只进行iou匹配, unconfirmed
          1: 已确认阶段, confirmed
          -1: 生命周期已结束, dead
        '''
        self.state = 0 
        self.hits = 1 # 单目标追踪到对象的次数
        self.kf = kalman_filter.KalmanFilter(measurement)


class MOTTracker(object):
    # 多目标追踪类
    def __init__(self, nn_budget=20, max_iou_distance=0.7, matching_time_depth=30, n_init=3, max_cosine_distance=0.2):
        self.budget = nn_budget # 控制Tracker对象收集到的feature数 
        self.max_iou_distance = max_iou_distance # iou匹配的阈值
        self.matching_time_depth = matching_time_depth # 参与匹配的时间框大小，此时间框之前的单目标最踪器不再参与匹配，即生命周期结束 
        self.n_init = n_init # 当track追踪到对象的次数达到n_init时，track状态由 unconfirmed 转化为 confirmed

        self.tracks = [] # 多个单目标追踪对象
        self._next_id = 1 # 下个新track的id
        self.distance = None
        self.max_cosine_distance = max_cosine_distance # feature匹配的阈值

    def predict(self):
        for i, track in enumerate(self.tracks):
            track.kf.predict()
            '''
            if i == 0:
                print(track.kf.state)
                print(track.kf.covariance)
                import pdb
                pdb.set_trace()
            '''
            self.tracks[i].time_since_update += 1

    def update(self, detections, budget_association_threshold=1.0, budget_detection_threshold=0):
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # 对于成功匹配，更新track状态
        for track_idx, detection_idx, cost in matches:
            self.tracks[track_idx].kf.correct(detections[detection_idx].to_xyah())
            '''
            if track_idx == 0:
                print(self.tracks[track_idx].kf.state)
                print(self.tracks[track_idx].kf.covariance)
                import pdb
                pdb.set_trace()
            '''
            if cost < budget_association_threshold and detections[detection_idx].confidence > budget_detection_threshold:
                # 满足条件的detection feature会被加入到track features中
                self.tracks[track_idx].features.append(detections[detection_idx].feature)
            self.tracks[track_idx].hits += 1
            self.tracks[track_idx].time_since_update = 0
            if self.tracks[track_idx].state == 0 and self.tracks[track_idx].hits >= self.n_init:
                # 满足条件，track状态由 unconfirmed 转化为 confirmed
                self.tracks[track_idx].state = 1
        # 对于未成功匹配，更新track状态
        for track_idx in unmatched_tracks:
            if self.tracks[track_idx].state == 0 or self.tracks[track_idx].time_since_update > self.matching_time_depth:
                '''
                满足两条件之一，track状态转化为 dead，生命周期结束
                1、track状态为 unconfirmed，且下次匹配未成功
                2、track距上次更新时长超过matching_time_depth
                '''
                self.tracks[track_idx].state = -1
        # 对于未成功匹配, 新建track
        for detection_idx in unmatched_detections:
            self.tracks.append(Tracker(detections[detection_idx].to_xyah(), self._next_id, detections[detection_idx].feature))
            '''
            if len(self.tracks) == 1:
                print(self.tracks[0].kf.state)
                print(self.tracks[0].kf.covariance)
                import pdb
                pdb.set_trace()
            '''
            self._next_id += 1
        # 丢弃处于dead状态的track
        self.tracks = [track for track in self.tracks if not track.state == -1]
        # 控制track的features的个数
        for track in self.tracks:
            track.features = track.features[-self.budget:]

    def _match(self, detections):
        '''
        match规则：
        1、处于confirmed状态的track，进行feature association
        2、处于confirmed状态但经过feature association未匹配成功的track，以及处于unconfirmed的track，进行 iou association
        '''
        confirmed_tracks = [i for i, track in enumerate(self.tracks) if track.state == 1]
        unconfirmed_tracks = [i for i, track in enumerate(self.tracks) if track.state == 0]

        unmatched_detections = list(range(len(detections)))

        # 按上次更新的时间由近到远级联匹配
        matches_a = []
        for level in range(self.matching_time_depth):
            if len(unmatched_detections) == 0:
                break

            track_indices_l = [k for k in confirmed_tracks if self.tracks[k].time_since_update == 1 + level]
            if len(track_indices_l) == 0:
                continue

            self.distance = self._distance_metric(detections, track_indices_l, unmatched_detections)
            self.distance[self.distance > self.max_cosine_distance] = self.max_cosine_distance + 1e-5
            matches_l, _, unmatched_detections = self._min_cost_match(detections, track_indices_l, unmatched_detections, self.max_cosine_distance)
            matches_a += matches_l
        unmatched_tracks_a = list(set(confirmed_tracks) - set(k for k, _, _ in matches_a))

        # Associate remaining tracks together with unconfirmed tracks using IOU.
        iou_track_candidates = unconfirmed_tracks + [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update == 1]
        unmatched_tracks_a = [
            k for k in unmatched_tracks_a if
            self.tracks[k].time_since_update != 1]

        matches_b = []
        unmatched_tracks_b = []
        if len(unmatched_detections) != 0 and len(iou_track_candidates) != 0:
            self.distance = self._iou_metric(detections, iou_track_candidates, unmatched_detections)
            self.distance[self.distance > self.max_iou_distance] = self.max_iou_distance + 1e-5
            matches_l, unmatched_tracks_l, unmatched_detections = self._min_cost_match(detections, iou_track_candidates, unmatched_detections, self.max_iou_distance)
            matches_b += matches_l
            unmatched_tracks_b += unmatched_tracks_l
        else:
            unmatched_tracks_b = iou_track_candidates

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        '''
        print([(self.tracks[a].track_id, b) for a, b, _ in matches_a])
        print([(self.tracks[a].track_id, b) for a, b, _ in matches_b])
        print([self.tracks[a].track_id for a in unmatched_tracks_a])
        print([self.tracks[b].track_id for b in unmatched_tracks_b])
        print(unmatched_detections)
        import pdb
        pdb.set_trace()
        '''

        return matches, unmatched_tracks, unmatched_detections

    def _min_cost_match(self, detections, track_indices, detection_indices, max_distance):
        # 使用匈牙利算法进行最小代价匹配
        indices = linear_assignment(self.distance)

        matches, unmatched_tracks, unmatched_detections = [], [], []
        for col, detection_idx in enumerate(detection_indices):
            if col not in indices[:, 1]:
                unmatched_detections.append(detection_idx)
        for row, track_idx in enumerate(track_indices):
            if row not in indices[:, 0]:
                unmatched_tracks.append(track_idx)
        for row, col in indices:
            track_idx = track_indices[row]
            detection_idx = detection_indices[col]
            if self.distance[row, col] > max_distance:
                unmatched_tracks.append(track_idx)
                unmatched_detections.append(detection_idx)
            else:
                matches.append((track_idx, detection_idx, self.distance[row, col]))
        return matches, unmatched_tracks, unmatched_detections
        

    # compute the distance between tracks and new detections
    def _distance_metric(self, detections, track_indices=None, detection_indices=None):
        '''
        feature distance metric的计算，使用track features（多个feature）与detection feature的最小距离
        '''
        if track_indices is None:
            track_indices = np.arange(len(self.tracks))
        if detection_indices is None:
            detection_indices = np.arange(len(detections))
        distance_metric = np.zeros((len(track_indices), len(detection_indices)))

        d_features = np.asarray([detections[indice].feature for indice in detection_indices])
        d_features = d_features / np.linalg.norm(d_features, axis=1, keepdims=True)
        for i, indice in enumerate(track_indices):
            track = self.tracks[indice]
            t_features = np.asarray(track.features)
            t_features = t_features / np.linalg.norm(t_features, axis=1, keepdims=True)
            distances = 1. - np.dot(t_features, d_features.T)
            #return distances.min(axis=0)
            '''
            feature distance metric的其他计算方法的尝试，
            将良好特征匹配的个数（track features中与detection feature距离小于0.01的个数）作为权重考虑
            '''
            '''
            num = np.zeros(distances.shape)
            num[distances < 0.01] = 1
            num = np.sum(num, axis=0)
            num[num==0] = 1
            distance_metric[i, :] = distances.min(axis=0) / num
            '''
            distance_metric[i, :] = distances.min(axis=0)
        
        gating_dim = 4
        gating_threshold = 9.4877
        measurements = np.asarray([detections[indice].to_xyah() for indice in detection_indices])
        for i, indice in enumerate(track_indices):
            track = self.tracks[indice]
            gating_distance = track.kf.gating_distance(measurements)
            '''
            if indice == 15:
                print(gating_distance)
                print(self.tracks[15].time_since_update)
                import pdb
                pdb.set_trace()
            '''
            #print(distance_metric)
            #print(i)
            #print(gating_distance)
            #print(gating_threshold)
            distance_metric[i, gating_distance > gating_threshold] = 1e+5

        return distance_metric


    # compute the iou between tracks and new detections
    def _iou_metric(self, detections, track_indices=None, detection_indices=None):
        if track_indices is None:
            track_indices = np.arange(len(self.tracks))
        if detection_indices is None:
            detection_indices = np.arange(len(self.detections))
        iou_metric = np.zeros((len(track_indices), len(detection_indices)))

        for row, track_idx in enumerate(track_indices):
            if self.tracks[track_idx].time_since_update > 1:
                iou_metric[row, :] = 1e+5 # max cost
                continue
            track_tlwh = self.tracks[track_idx].kf.state[:4].copy()
            track_tlwh[2] *= track_tlwh[3]
            track_tlwh[:2] -= track_tlwh[2:] / 2
            detections_tlwh = np.asarray([detections[i].tlwh for i in detection_indices])
            iou_metric[row, :] = 1. - self._iou(track_tlwh, detections_tlwh)

        return iou_metric

    def _iou(self, track_tlwh, detections_tlwh):
        track_tl = track_tlwh[:2]
        track_br = track_tl + track_tlwh[2:]
        #print(track_tlwh)
        #print(detections_tlwh)
        detections_tl = detections_tlwh[:, :2]
        detections_br = detections_tl + detections_tlwh[:, 2:]

        inters_tl = np.concatenate((np.maximum(track_tl[0], detections_tl[:, 0])[:, np.newaxis],
                                   np.maximum(track_tl[1], detections_tl[:, 1])[:, np.newaxis]), axis=1)
        
        inters_br = np.concatenate((np.minimum(track_br[0], detections_br[:, 0])[:, np.newaxis],
                                   np.minimum(track_br[1], detections_br[:, 1])[:, np.newaxis]), axis=1)
        inters_wh = np.maximum(0., inters_br - inters_tl)
        track_area = track_tlwh[2] * track_tlwh[3]
        detections_area = detections_tlwh[:, 2] * detections_tlwh[:, 3]
        inters_area = inters_wh[:, 0] * inters_wh[:, 1]
        return inters_area / (track_area + detections_area - inters_area)
