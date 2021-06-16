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

from sklearn.utils import shuffle

from PIL import Image
import gc
from dask import bag, diagnostics
from random import randrange

from . import augment

gc.collect()

# capturing details of subdirectories of desired folder
def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


# get file and folder details and move those details to a data frame for further use

def getFileDetails_df(path,folderList):
    totalImg=0
    df_overview=pd.DataFrame(columns=['Car Details','No of Images'])
    df_FileDetails=pd.DataFrame(columns=['file_name','fol_details','path'])
    for img in folderList:
        c1=len(os.listdir(os.path.join(path, img)))
        df_overview=df_overview.append({'Car Details': img ,'No of Images' : c1},ignore_index=True)
        for filename in os.listdir(os.path.join(path, img)):
            df_FileDetails=df_FileDetails.append({'file_name':filename,
                                                  'fol_details':img,
                                                  'path':os.path.join(path, img,filename)
                                                 },ignore_index=True)
        #print('{}   -->   {} training images'.format(img, len(os.listdir(os.path.join(path1, img)))))
        totalImg+=c1
 
    return totalImg, df_overview,df_FileDetails
    
def getImage(path,df,i):
    im = cv2.imread(str('{}/{}/{}'.format(path, df.fol_details[i],df.file_name[i])))
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im

def viewImageWithBounding(path,df,i):
    im = getImage(path,df,i)
    cv2.rectangle(im,( int(df.x1[i]),int(df.y1[i])), (int(df.x2[i]),int(df.y2[i])), (0,255,0), 2)
    #plt.imshow(im)
    st.image(im, use_column_width=True,clamp = True)
 
def returnImg(path,df,i):
    im = getImage(path,df,i)
    return cv2.rectangle(im,( int(df.x1[i]),int(df.y1[i])), (int(df.x2[i]),int(df.y2[i])), (0,255,0), 2)

 

def get_dims(file):
    img = cv2.imread(file)
    h,w = img.shape[:2]
    return h,w


def process(trainPath, testPath, annoTrainPath, annoTestPath):

    #setting for resizing all to uniform size and reducing memory utilization
    img_height=100
    img_width=100
    
    st.write('EDA started')
    #st.write('Train Selected folder:',trainPath)
    #st.write('Test Selected folder:',testPath)
    #st.write('Train Selected annotation file:',annoTrainPath)
    #st.write('Test Selected annotation file:',annoTestPath)
    #ask for train, test, annotations path
    
    #path1='data/Car Images-20210501T094840Z-001/Car Images/Train Images'
    #path2='data/Car Images-20210501T094840Z-001/Car Images/Test Images'
    #path3='data\Annotations-20210510T185520Z-001\Annotations'
    # fetching all directories list
    
    #dirList=next(os.walk(path1))[1]
    #dirList.sort()
    #dirList
    
    cat_Folder_list_train=get_immediate_subdirectories(trainPath)
    cat_Folder_list_test=get_immediate_subdirectories(testPath)
    
    with st.spinner('Importing train image data set to dataframe...'):
        trainImgCount, df_overviewTrain, df_trainFiles=getFileDetails_df(trainPath,cat_Folder_list_train)
        
    st.write('Total train images:',trainImgCount)
    with st.spinner('Importing test image dataset to dataframe...'):
        testImgCount, df_overviewTest, df_testFiles=getFileDetails_df(testPath,cat_Folder_list_test)
        
 
    st.write('Total test images:',testImgCount)
    
    
    st.write('Distribution plot for number of images in each class, blue for Train & green for Test data set')
    fig = plt.figure()
    ax = sns.distplot(df_overviewTrain['No of Images'],color='b')
    ax = sns.distplot(df_overviewTest['No of Images'],color='g')
    fig.add_subplot(ax)
    plt.show()
    st.pyplot(fig)
    
    df_annotations_train = pd.read_csv(annoTrainPath)
    df_annotations_test  = pd.read_csv(annoTestPath)
    
    cols=['file_name','x1','y1','x2','y2','class_Id']
    df_annotations_train.columns=cols
    df_annotations_test.columns=cols
    
    
    df_train = pd.merge(df_trainFiles,df_annotations_train,on='file_name')
    df_test  = pd.merge(df_testFiles,df_annotations_test,on='file_name')
    
    st.write('Training data frame created')
    st.dataframe(df_train.head())
    
    st.write('Visalizing image with bounding boxes from train data set')
    viewImageWithBounding(trainPath,df_train,9)
    
    st.write('Visalizing image with bounding boxes from test data set')
    viewImageWithBounding(testPath,df_test,9)
    
    
    
    filelist=[]

    filelist = df_trainFiles.apply(lambda row : os.path.join(trainPath, row['fol_details'],row['file_name']), axis = 1)
    #getImage(path,df,i)
    
    #filelist = [filepath + f if f.endswith(".png") for f in os.listdir(filepath)]
    #print(filelist)
    
    
    dimsbag = bag.from_sequence(filelist).map(get_dims)
    
    #with diagnostics.ProgressBar():
    with st.spinner('Computing dimension details...'): 
        dims = dimsbag.compute()
        
    dim_df = pd.DataFrame(dims, columns=['height', 'width'])
    sizes = dim_df.groupby(['height', 'width']).size().reset_index().rename(columns={0:'count'})
    
    
    st.write('Max and Min count for size of image:',max(sizes['count']), min(sizes['count']))
    st.write('Max and Min height of images:', max(sizes['height']), min(sizes['height']))
    st.write('Max and Min width of images:', max(sizes['width']), min(sizes['width']))

    st.write('Image size having highest occurance: ', sizes.loc[sizes['count']==max(sizes['count'])])
    
    #fig = plt.figure()
    fig, ax = plt.subplots()
    sizes.plot.scatter(x='height', y='width',c='count',s=sizes['count'],colormap='viridis', ax=ax)
    #fig.add_subplot(ax)
    plt.show()
    st.pyplot(fig)
    
    num = randrange(len(df_train))
    st.subheader('Image Augmentation')
    #st.markdown('#### Original Image:')
    #viewImageWithBounding(trainPath,df_train,num)
    
    st.markdown('#### Resized Image with letter box keeping aspect ratio, Original vs Augmented:')
    img,bboxes=augment.loadImgWithBbox(trainPath,df_train,num)
    img,bboxes=augment.resize(img,bboxes,(300,300))
    #augment.viewUpdatedImg(img,bboxes)
    st.image([returnImg(trainPath,df_train,num),augment.viewImg(img,bboxes)], width=300)
    
    num = randrange(len(df_train))
    st.markdown('#### Resized Image without letter box not keeping aspect ratio, Original vs Augmented:')
    img,bboxes=augment.loadImgWithBbox(trainPath,df_train,num)
    img,bboxes=augment.resizeWithoutLetterBox(img,bboxes,(300,300))
    #augment.viewUpdatedImg(img,bboxes)
    st.image([returnImg(trainPath,df_train,num),augment.viewImg(img,bboxes)], width=300)
         
    num = randrange(len(df_train))   
    st.markdown('#### Horizontal Flip, Original vs Augmented:')
    img,bboxes=augment.loadImgWithBbox(trainPath,df_train,num)
    img,bboxes=augment.horzFlip(img,bboxes)
    #augment.viewUpdatedImg(img,bboxes)
    st.image([returnImg(trainPath,df_train,num),augment.viewImg(img,bboxes)], width=300)
          
    num = randrange(len(df_train))  
    st.markdown('#### Scale Image, Original vs Augmented:')
    img,bboxes=augment.loadImgWithBbox(trainPath,df_train,num)
    img,bboxes=augment.scaleImage(img,bboxes,0.4,True)
    #augment.viewUpdatedImg(img,bboxes)
    st.image([returnImg(trainPath,df_train,num),augment.viewImg(img,bboxes)], width=300)
           
    num = randrange(len(df_train))
    st.markdown('#### Translate Image, Original vs Augmented:')
    img,bboxes=augment.loadImgWithBbox(trainPath,df_train,num)
    img,bboxes=augment.translateImage(img,bboxes,0.4,True)
    #augment.viewUpdatedImg(img,bboxes)
    st.image([returnImg(trainPath,df_train,num),augment.viewImg(img,bboxes)], width=300)
           
    num = randrange(len(df_train))
    st.markdown('#### Rotate Image, Original vs Augmented:')
    img,bboxes=augment.loadImgWithBbox(trainPath,df_train,num)
    img,bboxes=augment.rotateImageBBox(img,bboxes,30.0)
    #augment.viewUpdatedImg(img,bboxes)
    st.image([returnImg(trainPath,df_train,num),augment.viewImg(img,bboxes)], width=300)
            
    num = randrange(len(df_train))
    st.markdown('#### Shear Image, Original vs Augmented:')
    img,bboxes=augment.loadImgWithBbox(trainPath,df_train,num)
    img,bboxes=augment.shearImage(img,bboxes,0.7)
    #augment.viewUpdatedImg(img,bboxes)
    st.image([returnImg(trainPath,df_train,num),augment.viewImg(img,bboxes)], width=300)

    return df_train,df_test