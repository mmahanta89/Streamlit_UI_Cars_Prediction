# -*- coding: utf-8 -*-
"""
Created on Sat May 29 18:38:01 2021

@author: Manoj-PC
"""
from __future__ import absolute_import
import streamlit as st
import sys
import numpy as np
import pandas as pd
import pickle
import os
import seaborn as sns
import matplotlib.pyplot as plt

import cv2
import time
import itertools
import random
from PIL import Image
import gc


def letterbox_image(img, inp_dim):
    '''resize image with unchanged aspect ratio using padding'''
    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    #print(new_h,new_w)
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 0)
 
    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas

def viewUpdatedImg(image,bbox):
    im=cv2.rectangle(image.copy(), (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,180,0), 2)
    #plt.imshow(im)
    st.image(im, use_column_width=True,clamp = True)


def viewImg(image,bbox):
    return cv2.rectangle(image.copy(), (bbox[0],bbox[1]), (bbox[2],bbox[3]), (0,180,0), 2)


    
    
def loadImage(path,df,i):
    im = cv2.imread(str('{}/{}/{}'.format(path, df.fol_details[i],df.file_name[i])))[:,:,::-1]
    #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)#change color space
    #cv2.rectangle(im,( int(df.x1[i]),int(df.y1[i])), (int(df.x2[i]),int(df.y2[i])), (0,255,0), 2)
    return im

def loadImgWithBbox(path, df,i):
    img=loadImage(path, df,i)
    bboxes=np.array([df.x1[i],df.y1[i],df.x2[i],df.y2[i]])
    return img, bboxes    
    


def resize(img, bboxes,inp_dim):
 
    
    w,h = img.shape[1], img.shape[0]
    img = letterbox_image(img, inp_dim)


    scale = min(inp_dim[1]/h, inp_dim[0]/w)
    bboxes[:4] = bboxes[:4]  * (scale)

    new_w = scale*w
    new_h = scale*h
    inp_dim = inp_dim   

    del_h = (inp_dim[1] - new_h)/2
    del_w = (inp_dim[0] - new_w)/2

    add_matrix = np.array([[del_w, del_h, del_w, del_h]]).astype(int)

    bboxes[:4] = bboxes[:4] + add_matrix

    img = img.astype(np.uint8)

    return img, bboxes



def resizeWithoutLetterBox(img, bboxes,inp_dim):
 
    
    w,h = img.shape[1], img.shape[0]

    x_scale = inp_dim[0]/w
    y_scale = inp_dim[1]/h
    #print(bboxes)
    #print(x_scale,y_scale)
    img = cv2.resize(img,(inp_dim))
    img = np.array(img)
    
    
    x = int(np.ceil(bboxes[0]*x_scale))
    y = int(np.ceil(bboxes[1]*y_scale))
    xmax= int(np.ceil(bboxes[2]*(x_scale)))
    ymax= int(np.ceil(bboxes[3]*y_scale))
    

    bboxes[:4] =  np.array([[x, y, xmax, ymax]]).astype(int)
    #print(bboxes)
   

    img = img.astype(np.uint8)

    return img, bboxes


def horzFlip(img, bboxes):
    #print(bboxes)
    img_center = np.array(img.shape[:2])[::-1]/2
    img_center = np.hstack((img_center, img_center))

    img =  img[:,::-1,:]
    bboxes[[0,2]] = bboxes[[0,2]] + 2*(img_center[[0,2]] - bboxes[[0,2]])
    #print(bboxes)
    box_w = abs(bboxes[0] - bboxes[2])
    bboxes[0] = bboxes[0] - box_w
    bboxes[2] = bboxes[2] + box_w
    #print(bboxes)
    
    return img, bboxes

def clip_box(bbox, clip_box):

    #ar_ = (bbox_area(bbox))
    bbox[0] = np.maximum(bbox[0], clip_box[0])
    bbox[1] = np.maximum(bbox[1], clip_box[1])
    bbox[2] = np.minimum(bbox[2], clip_box[2])
    bbox[3] = np.minimum(bbox[3], clip_box[3])
    #print(x_min)
    #bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[4:]))
    #print(bbox)
    #delta_area = ((ar_ - bbox_area(bbox))/ar_)
    
    #mask = (delta_area < (1 - alpha)).astype(int)
    
    #bbox = bbox[mask == 1,:]


    return bbox

def bbox_area(bbox):
    return (bbox[2] - bbox[0])*(bbox[3] - bbox[1])

def scaleImage(img,bboxes,scale,diff):
    
    if (type(scale) == tuple):
        assert len(scale) == 2, "Invalid range"
        assert scale[0] > -1, "Scale factor can't be less than -1"
        assert scale[1] > -1, "Scale factor can't be less than -1"
    else:
        assert scale > 0, "Please input a positive float"
        scale = (max(-1, -scale), scale)
        
     
    img_shape = img.shape
        
    if diff:
        scale_x = random.uniform(*scale)
        scale_y = random.uniform(*scale)
    else:
        scale_x = random.uniform(*scale)
        scale_y = scale_x

    resize_scale_x = 1 + scale_x
    resize_scale_y = 1 + scale_y

    img=  cv2.resize(img, None, fx = resize_scale_x, fy = resize_scale_y)

    bboxes[:4] = bboxes[:4]  * [resize_scale_x, resize_scale_y, resize_scale_x, resize_scale_y]
    
    canvas = np.zeros(img_shape, dtype = np.uint8)
    
    y_lim = int(min(resize_scale_y,1)*img_shape[0])
    x_lim = int(min(resize_scale_x,1)*img_shape[1])

    canvas[:y_lim,:x_lim,:] =  img[:y_lim,:x_lim,:]

    img = canvas
    
    bboxes = clip_box(bboxes, [0,0,1 + img_shape[1], img_shape[0]])
    
    return img, bboxes


def translateImage(img,bboxes,translate,diff):
    
    if type(translate) == tuple:
        assert len(translate) == 2, "Invalid range"  
        assert translate[0] > 0 and translate[0] < 1
        assert translate[1] > 0 and translate[1] < 1
    else:
        assert translate > 0.0 and translate < 1.0
        translate = (-translate, translate)
    
    img_shape = img.shape
        
    translate_factor_x = random.uniform(*translate)
    translate_factor_y = random.uniform(*translate)
    
    if not diff:
        translate_factor_y = translate_factor_x
        
    canvas = np.zeros(img_shape).astype(np.uint8)
    
    corner_x = int(translate_factor_x*img.shape[1])
    corner_y = int(translate_factor_y*img.shape[0])
    
    orig_box_cords =  [max(0,corner_y), 
                       max(corner_x,0), 
                       min(img_shape[0], corner_y + img.shape[0]), 
                       min(img_shape[1],corner_x + img.shape[1])
                      ]
    
    mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]), max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]),:]
    canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3],:] = mask
    img = canvas
    
    bboxes[:4] = bboxes[:4] + [corner_x, corner_y, corner_x, corner_y]
    
    bboxes = clip_box(bboxes, [0,0,img_shape[1], img_shape[0]])
    
    return img, bboxes





#rotate_im
def rotate_im(image, angle):

    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    #center = tuple(np.array(image.shape)[:2]/2)

    # print(angle[0],cX,cY)
    
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    image = cv2.warpAffine(image, M, (nW, nH))

#    image = cv2.resize(image, (w,h))
    return image

def get_corners(bboxes):
    
    width = (bboxes[2] - bboxes[0]).reshape(-1,1)
    height = (bboxes[3] - bboxes[1]).reshape(-1,1)
    
    x1 = bboxes[0].reshape(-1,1)
    y1 = bboxes[1].reshape(-1,1)
    
    x2 = x1 + width
    y2 = y1 
    
    x3 = x1
    y3 = y1 + height
    
    x4 = bboxes[2].reshape(-1,1)
    y4 = bboxes[3].reshape(-1,1)
    
    corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
    
    return corners

def rotate_box(corners,angle,  cx, cy, h, w):


    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M,corners.T).T
    
    calculated = calculated.reshape(-1,8)
    
    return calculated

def get_enclosing_box(corners):

    x_ = corners[[0,2,4,6]]
    y_ = corners[[1,3,5,7]]
    
    
    xmin = np.min(x_).reshape(-1,1)
    ymin = np.min(y_).reshape(-1,1)
    xmax = np.max(x_).reshape(-1,1)
    ymax = np.max(y_).reshape(-1,1)
    
    #print(xmin,xmin[0])
    
    final = np.hstack((xmin[0], ymin[0], xmax[0], ymax[0],corners[8:]))
    
    return final

def rotateImageBBox(img,bboxes,angle):
    
    
    if type(angle) == tuple:
        assert len(angle) == 2, "Invalid Range"
    else:
        angle = (-angle,angle)
    
    angle = random.uniform(*angle)
    
    w,h = img.shape[1], img.shape[0]
    cx, cy = w//2, h//2
    
    img = rotate_im(img, angle)

    corners = get_corners(bboxes)

    corners = np.hstack((corners[0], bboxes[4:]))


    corners[:8] = rotate_box(corners[:8], angle, cx, cy, h, w)

    new_bbox = get_enclosing_box(corners)


    scale_factor_x = img.shape[1] / w

    scale_factor_y = img.shape[0] / h

    img = cv2.resize(img, (w,h))

    new_bbox[:4] = new_bbox[:4]  / [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 

    bboxes  = new_bbox

    bboxes = clip_box(bboxes, [0,0,w, h])

    return img, bboxes

def shearImage(img,bboxes,shear_factor):
 
    if type(shear_factor) == tuple:
        assert len(shear_factor) == 2, "Invalid range for scaling factor"   
    else:
        shear_factor = (-shear_factor, shear_factor)
        
    shear_factor = random.uniform(*shear_factor)          
     
    w,h = img.shape[1], img.shape[0]

    if shear_factor < 0:
        img, bboxes = horzFlip(img,bboxes)

    M = np.array([[1, abs(shear_factor), 0],[0,1,0]])

    nW =  img.shape[1] + abs(shear_factor*img.shape[0])

    bboxes[[0,2]] = bboxes[[0,2]] + ((bboxes[[1,3]]) * abs(shear_factor) ).astype(int) 


    img = cv2.warpAffine(img, M, (int(nW), img.shape[0]))

    if shear_factor < 0:
        img, bboxes = horzFlip(img,bboxes)

    img = cv2.resize(img, (w,h))

    scale_factor_x = nW / w

    bboxes[:4] = bboxes[:4] / [scale_factor_x, 1, scale_factor_x, 1] 


    return img, bboxes
