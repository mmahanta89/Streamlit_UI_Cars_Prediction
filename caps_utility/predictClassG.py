# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 19:52:12 2021

@author: Manoj-PC
"""

from __future__ import division
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 
import tensorflow as tf
import cv2
import keras
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import array_to_img                                     
from PIL import Image

from . import eda_visualization as eda

from keras.models import load_model
img_size = 224

def predictImg(uploaded_file, weights, showImg = True):
    
    model = tf.keras.applications.ResNet50(include_top=False,input_shape=(224,224,3))
    
    x1 = model.output
    x2 = tf.keras.layers.GlobalAveragePooling2D()(x1)
    x3 = tf.keras.layers.Dropout(0.2)(x2)
    x4 = tf.keras.layers.BatchNormalization()(x3)
    x5 = tf.keras.layers.Dense(200)(x4)
    x6 = tf.keras.layers.BatchNormalization()(x5)
    
    
    prediction = tf.keras.layers.Dense(196,activation='softmax')(x6)
        
    final_model = tf.keras.models.Model(inputs = model.input,outputs=prediction)
    st.write("Model created")  
    
    final_model = load_model(weights)
    st.write('Weights Loaded')
    
    import pickle

    with open('carClassDict.pickle', 'rb') as handle:
        dictClass = pickle.load(handle)
    if showImg:
        progress_bar = st.progress(0.0)
    
    with st.spinner('Predictng car calsses ... wait for it...'):
        for idx, img in enumerate(uploaded_file):
            if showImg:
                progress_bar.progress((idx + 0.1) / len(uploaded_file))
            img = Image.open(img)
            img_resized = img.resize((img_size, img_size))
            img_array = tf.keras.preprocessing.image.img_to_array(img_resized).astype('uint8')
            
            input_array = np.expand_dims(img_array, axis=0)
            input_array = tf.keras.applications.resnet50.preprocess_input(input_array)
            label_pred = final_model.predict(input_array)
            pred_class = dictClass.get(str(np.argmax(label_pred)))
            
            
            st.write('Predicted Label :', pred_class)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            plt.imshow(img)
            if showImg:
                st.image(img, use_column_width=True,clamp = True)
     
    if showImg:
        progress_bar.empty()
    
    
def predictTestImg(test_path, annoTestPath, image_nums, weights):
    st.write('Predicting from test data set')
    with st.spinner('Predictng car calsses ... wait for it...'):
        test_df = pd.read_csv( annoTestPath ,header = None,names = ['file','x_min','y_min','x_max','y_max','class'] )
           
        
        # Test images with its classification
        classification2=[]
        #category2 = os.listdir(test_path)
        category2 = eda.get_immediate_subdirectories(test_path)
        for category in category2:
            for i in range(0,len(os.listdir(os.path.join(test_path,category)))):
                image = os.listdir(os.path.join(test_path,category))[i]
                classification2.append([image,category])
        
        
        #converting list to dataframe
        test_classification = pd.DataFrame(classification2,columns=['file','classification'])
        
        # Merging the data frames
        test_df_final = test_df.merge(test_classification, on="file", how = 'inner')
        
        test_df_final['file_path']=test_path+'/'+test_df_final['classification']+'/'+test_df_final['file']
        
        # Adding the height & width field with the dataframe
        test_df_final['width'] = 0
        test_df_final['height'] = 0
        
        #dictClass = test_df_final.set_index('class')['classification'].to_dict()
        
        import pickle
        # with open('carClassDict.pickle', 'wb') as handle:
        #     pickle.dump(dictClass, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #st.write(dictClass)
        with open('carClassDict.pickle', 'rb') as handle:
            dictClass = pickle.load(handle)
        
        
        # extracting the image width & height and loading to dataframe
        for i in range(0,test_df_final['file_path'].shape[0]):
            image = load_img(test_df_final['file_path'][i])
            test_df_final['width'][i]=image.size[0]
            test_df_final['height'][i]=image.size[1]
        
        
        if image_nums >= len(test_df_final):
            st.write('Image number provided is higher then available images, chosing image randomly')
            st.write('Total test images: ', len(test_df_final))
            image_nums = np.random.randint(0,test_df_final.shape[0])
            st.write('Random image seleccted: ',image_nums)
        
        img = Image.open(test_df_final.loc[image_nums,'file_path'])
        #left = test_df_final.loc[image_nums,'x_min']
        #right = test_df_final.loc[image_nums,'x_max']
        #top = test_df_final.loc[image_nums,'y_min']
        #bottom = test_df_final.loc[image_nums,'y_max']
        
        #img = img.crop((left, top, right, bottom)) 
        img_resized = img.resize((img_size, img_size))
        img_array = tf.keras.preprocessing.image.img_to_array(img_resized).astype('uint8')
        
        input_array = np.expand_dims(img_array, axis=0)      
        
          #3. Normalize image data
        input_array = tf.keras.applications.resnet50.preprocess_input(input_array)
    
        
         
        model = tf.keras.applications.ResNet50(include_top=False,input_shape=(224,224,3))
        
        x1 = model.output
        x2 = tf.keras.layers.GlobalAveragePooling2D()(x1)
        x3 = tf.keras.layers.Dropout(0.2)(x2)
        x4 = tf.keras.layers.BatchNormalization()(x3)
        x5 = tf.keras.layers.Dense(200)(x4)
        x6 = tf.keras.layers.BatchNormalization()(x5)
        
        
        prediction = tf.keras.layers.Dense(196,activation='softmax')(x6)
            
        final_model = tf.keras.models.Model(inputs = model.input,outputs=prediction)
        st.write("Model created")  
        
        final_model = load_model(weights)
        st.write('Weights Loaded')
        
        
        label_pred = final_model.predict(input_array)
        pred_class1 = np.argmax(label_pred)
        #st.write('Predicted Number :', pred_class1)
        pred_class = dictClass.get(str(pred_class1))
        act_class = dictClass.get(str(test_df_final.loc[image_nums, 'class']))
        
        st.write('Predicted Label :', pred_class)
        st.write('Actual lable :', act_class)
        
        
        plt.imshow(img)
        st.image(img, use_column_width=True,clamp = True)
        
   