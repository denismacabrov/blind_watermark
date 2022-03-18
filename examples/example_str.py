#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# embed string
import numpy as np
from blind_watermark import WaterMark
from blind_watermark import att
from blind_watermark.recover import estimate_crop_parameters, recover_crop

import cv2

bwm = WaterMark(password_img=1, password_wm=1)
bwm.read_img('pic/ori_img.jpeg')
wm = 'the quick brown fox jumps over the lazy dog'
bwm.read_wm(wm, mode='str')
bwm.embed('output/embedded.png')

len_wm = len(bwm.wm_bit)  # You must set length to decode watermark
print('Put down the length of wm_bit {len_wm}'.format(len_wm=len_wm))

ori_img_shape = cv2.imread('pic/ori_img.jpeg').shape[:2]  # Sometimes you need to know shape to recover after attack

# %% Watermark
bwm1 = WaterMark(password_img=1, password_wm=1)
wm_extract = bwm1.extract('output/embedded.png', wm_shape=len_wm, mode='str')
print("Recovering watermark without attack:", wm_extract)

assert wm == wm_extract, 'Extracted watermark didnt match original'

# %% Screenshot attack = framing + resize + known attack parameters

loc = ((0.1, 0.1), (0.5, 0.5))
resize = 0.7
att.cut_att(input_filename='output/embedded.png', output_file_name='output/screenshot_attack.png', loc=loc, resize=resize)

bwm1 = WaterMark(password_wm=1, password_img=1)
wm_extract = bwm1.extract('output/screenshot_attack.png', wm_shape=len_wm, mode='str')
print("Screenshot attack ={loc}，zoom={resize}，extract results, knowing the parameters:".format(loc=loc, resize=resize), wm_extract)
assert wm == wm_extract, 'Extracted watermark didnt match original'

# %% Screenshot attack = framing + resize + unknown attack parameters
loc_r = ((0.1, 0.1), (0.7, 0.6))
scale = 0.7
_, (x1, y1, x2, y2) = att.cut_att2(input_filename='output/embedded.png', output_file_name='output/screenshot_attack2.png',
                                   loc_r=loc_r, scale=scale)
print(f'Crop attack\'s real parameters: x1={x1},y1={y1},x2={x2},y2={y2}')

# estimate crop attack parameters:
(x1, y1, x2, y2), image_o_shape, score, scale_infer = estimate_crop_parameters(original_file='output/embedded.png',
                                                                               template_file='output/screenshot_attack2.png',
                                                                               scale=(0.5, 2), search_num=200)

print(f'Crop attack\'s estimate parameters: x1={x1},y1={y1},x2={x2},y2={y2}. score={score}')

# recover from attack:
recover_crop(template_file='output/screenshot_attack2.png', output_file_name='output/screenshot_attack2_recovered.png',
             loc=(x1, y1, x2, y2), image_o_shape=image_o_shape)

bwm1 = WaterMark(password_wm=1, password_img=1)
wm_extract = bwm1.extract('output/screenshot_attack2_recovered.png', wm_shape=len_wm, mode='str')
print("Extracting results, without knowing parameters of attack:", wm_extract)
assert wm == wm_extract, 'Extracted watermark didnt match original'

# %% Vertical cut
r = 0.3
att.cut_att_width(input_filename='output/embedded.png', output_file_name='output/horizontal_crop.png', ratio=r)
att.anti_cut_att(input_filename='output/horizontal_crop.png', output_file_name='output/horizontal_crop_recovered.png',
                 origin_shape=ori_img_shape)

# Extracting watermark
bwm1 = WaterMark(password_wm=1, password_img=1)
wm_extract = bwm1.extract('output/horizontal_crop_recovered.png', wm_shape=len_wm, mode='str')
print(f"Horizontal crop attack with r={r}. Extracting results:", wm_extract)

assert wm == wm_extract, 'Extracted watermark didnt match original'

# %% horizontal cut
r = 0.4
att.cut_att_height(input_filename='output/embedded.png', output_file_name='output/vertical_crop.png', ratio=r)
att.anti_cut_att(input_filename='output/vertical_crop.png', output_file_name='output/vertical_crop_recovered.png',
                 origin_shape=ori_img_shape)

# extract:
bwm1 = WaterMark(password_wm=1, password_img=1)
wm_extract = bwm1.extract('output/vertical_crop_recovered.png', wm_shape=len_wm, mode='str')
print(f"Vertical crop attack with r={r}. Extracting results:", wm_extract)

assert wm == wm_extract, 'Extracted watermark didnt match original'
# %% salt-pepper attack
ratio = 0.05
att.salt_pepper_att(input_filename='output/embedded.png', output_file_name='output/salt_pepper.png', ratio=ratio)
# ratio是椒盐概率

# extract
wm_extract = bwm1.extract('output/salt_pepper.png', wm_shape=len_wm, mode='str')
print(f"Salt&pepper attack with ratio={ratio}. Extracting results:", wm_extract)
assert np.all(wm == wm_extract), 'Extracted watermark didnt match original'

# %% rotate attack
angle = 60
att.rot_att(input_filename='output/embedded.png', output_file_name='output/rotate.png', angle=angle)
att.rot_att(input_filename='output/rotate.png', output_file_name='output/rotate_recovered.png', angle=-angle)

# extract watermark
bwm1 = WaterMark(password_wm=1, password_img=1)
wm_extract = bwm1.extract('output/rotate_recovered.png', wm_shape=len_wm, mode='str')
print(f"Rotate attack with angle={angle}. Extracting results:", wm_extract)
assert wm == wm_extract, 'Extracted watermark didnt match original'

# %%遮挡攻击
n = 60
att.shelter_att(input_filename='output/embedded.png', output_file_name='output/occlusion.png', ratio=0.1, n=n)

# extract
bwm1 = WaterMark(password_wm=1, password_img=1)
wm_extract = bwm1.extract('output/occlusion.png', wm_shape=len_wm, mode='str')
print(f"Occlusion {n} times attack. Extracting results:", wm_extract)
assert wm == wm_extract, 'Extracted watermark didnt match original'

# %%缩放攻击
att.resize_att(input_filename='output/embedded.png', output_file_name='output/resize.png', out_shape=(400, 300))
att.resize_att(input_filename='output/resize.png', output_file_name='output/resize_recovered.png',
               out_shape=ori_img_shape[::-1])
# out_shape 是分辨率，需要颠倒一下

bwm1 = WaterMark(password_wm=1, password_img=1)
wm_extract = bwm1.extract('output/resize_recovered.png', wm_shape=len_wm, mode='str')
print("Resize attack. Extracting results:", wm_extract)
assert np.all(wm == wm_extract), 'Extracted watermark didnt match original'
# %%

att.bright_att(input_filename='output/embedded.png', output_file_name='output/bright.png', ratio=0.9)
att.bright_att(input_filename='output/bright.png', output_file_name='output/bright_recovered.png', ratio=1.1)
wm_extract = bwm1.extract('output/bright_recovered.png', wm_shape=len_wm, mode='str')

print("Brightness attack. Extracting results:", wm_extract)
assert np.all(wm == wm_extract), 'Extracted watermark didnt match original'
