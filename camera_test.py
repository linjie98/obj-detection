#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author   : linjie
import os
import cv2
from base_camera import BaseCamera
from models.experimental import attempt_load
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torchvision
import numpy as np
import argparse
from utils.datasets import *
from utils.utils import *
from utils.torch_utils import select_device, load_classifier, time_synchronized
from flask import Flask,url_for
from log import Logger

from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config

class Camera(BaseCamera):

    #video_source = 'people.mp4'
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    video_source = 'people.mp4'
    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE'):
            print('走了吗')
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()
        print('走了')
    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def bbox_rel(*xyxy):
        """" Calculates the relative bounding box from absolute pixel values. """
        bbox_left = min([xyxy[0].item(), xyxy[2].item()])
        bbox_top = min([xyxy[1].item(), xyxy[3].item()])
        bbox_w = abs(xyxy[0].item() - xyxy[2].item())
        bbox_h = abs(xyxy[1].item() - xyxy[3].item())
        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)
        w = bbox_w
        h = bbox_h
        return x_c, y_c, w, h

    @staticmethod
    def compute_color_for_labels(label):
        """
        Simple function that adds fixed color depending on the class
        """
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in Camera.palette]
        return tuple(color)
    # @staticmethod
    # def people_appeal1():
    #     a = 'people'
    #     yield a
    @staticmethod
    def draw_boxes(img, bbox, cls_names, scores, identities=None, offset=(0,0)):
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]
            # box text and bar
            id = int(identities[i]) if identities is not None else 0
            color = Camera.compute_color_for_labels(id)
            label = '%d %s %d' % (id, cls_names[i], scores[i])
            label += '%'
            print("{0}号人物出现！========================================".format(id))
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2, 2)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
            cv2.rectangle(
                img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (x1, y1 +
                                     t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 2, [255, 255, 255], 2)
        return img
    @staticmethod
    def frames():
        logger = Logger()
        print('初始化过了，。。。。。')
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera.')

        out, weights, imgsz = \
        '.inference/output', 'weights/yolov5s.pt', 640
        source = "people.mp4"
        # print(source)
        # print(type(source))
        #webcam = source.isnumeric()
        # print('看看webcam:{0}'.format(webcam))
        '''
        初始化deepsort
        '''
        # initialize deepsort
        # cfg = get_config()
        # cfg.merge_from_file('deep_sort_pytorch/configs/deep_sort.yaml')
        # deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
        #                     max_dist=cfg.DEEPSORT.MAX_DIST, min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        #                     nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
        #                     max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        #                     max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET,
        #                     use_cuda=True)




        device = torch_utils.select_device()
        # print(weights)
        # print(os.getcwd())
        if os.path.exists(out):
            shutil.rmtree(out)  # delete output folder
        os.makedirs(out)  # make new output folder
        #shutil.rmtree(out)
        # Load model
        # google_utils.attempt_download(weights)
        # model = torch.load(weights, map_location=device)['model']
        model = attempt_load(weights, map_location=device)  # load FP32 model
        model.to(device).eval()

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
            modelc.to(device).eval()

        # Half precision
        half = False and device.type != 'cpu'
        # print('half = ' + str(half))

        if half:
            model.half()

        # Set Dataloader
        vid_path, vid_writer = None, None
        # #if webcam:
        # view_img = True
        # cudnn.benchmark = True  # set True to speed up constant image size inference
        # dataset = LoadStreams(source, img_size=imgsz)
        # else:
        # save_img = True
        # #     # 如果检测视频的时候想显示出来，可以在这里加一行view_img = True
        # view_img = True
        #     dataset = LoadImages(source, img_size=imgsz)
        # vid_path, vid_writer = None, None
        dataset = LoadImages(source, img_size=imgsz)
        #dataset = LoadStreams(source, img_size=imgsz)
        # print('看看dataset:{0}'.format(dataset))
        names = model.names if hasattr(model, 'names') else model.modules.names
        # print('----')
        # print(names)
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        # Run inference
        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
        for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            # Inference
            t1 = torch_utils.time_synchronized()
            pred = model(img, augment=False)[0]

            # Apply NMS
            pred = non_max_suppression(pred, 0.4, 0.5,
                               fast=True, classes=None, agnostic=False)
            t2 = torch_utils.time_synchronized()

            # Apply Classifier
            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)
            people_coords = []
            for i, det in enumerate(pred):  # detections per image
                #p, s, im0 = path, '', im0s
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                #save_path = str(Path(out) / Path(p).name)
                s += '%gx%g ' % img.shape[2:]  # print string
                #gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh

                if det is not None and len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    #for c in det[:, -1].unique():  #probably error with torch 1.5
                    for c in det[:, -1].detach().unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += '%g %s, ' % (n, names[int(c)])  # add to string

                    # --- linjie
                    # bbox_xywh = []
                    # confs = []
                    # clses = []

                    # for *xyxy, conf, cls in det:
                    #     label = '%s %.2f' % (names[int(cls)], conf)
                    #     plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                        # Write results
                    for *xyxy, conf, cls in reversed(det):
                            # -- linjie deepsort
                            # x_c, y_c, bbox_w, bbox_h = Camera.bbox_rel(*xyxy)
                            # obj = [x_c, y_c, bbox_w, bbox_h]
                            # bbox_xywh.append(obj)
                            # confs.append([conf.item()])
                            # clses.append([cls.item()])

                            label = '%s %.2f' % (names[int(cls)], conf)
                            print('看看这次打的标签：{0}'.format(label))

                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)
                            # 判断标签是否为人 --linjie
                            if label is not None:
                                if (label.split())[0] == 'person':
                                    logger.info('当前进程：{0}.遇到了人'.format(os.getpid()))
                                    #print('标签是人')
                                    # distancing(people_coords, im0, dist_thres_lim=(200, 250))
                                    # people_coords.append(xyxy)
                                    plot_one_box(xyxy, im0, line_thickness=3)
                                    plot_dots_on_people(xyxy, im0)



                print('%sDone. (%.3fs)' % (s, t2 - t1))
                # Stream results
                # if view_img:
                #     cv2.imshow(p, im0)
                #     if cv2.waitKey(1) == ord('q'):  # q to quit
                #         raise StopIteration
            yield cv2.imencode('.jpg', im0)[1].tobytes()


