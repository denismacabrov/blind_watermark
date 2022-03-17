#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2

from blind_watermark import WaterMark

bwm = WaterMark(password_wm=1, password_img=1)
# Read source
bwm.read_img(filename='pic/ori_img.jpeg')
# Read watermark
bwm.read_wm('pic/watermark.png')
# Embed blind watermark
bwm.embed('output/embedded.png')
wm_shape = cv2.imread('pic/watermark.png', flags=cv2.IMREAD_GRAYSCALE).shape

# %% Extracting watermark

bwm1 = WaterMark(password_wm=1, password_img=1)
# You must set width and height of extracted watermark in wm_shape
bwm1.extract('output/embedded.png', wm_shape=wm_shape, out_wm_name='output/wm_extracted.png', mode='img')
