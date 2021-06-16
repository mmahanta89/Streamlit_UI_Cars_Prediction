# -*- coding: utf-8 -*-
"""
Created on Sat Jun  5 19:32:22 2021

@author: Manoj-PC
"""
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

from . import eda_visualization as eda


def trainModel(trainPath, testPath, annoTrainPath, annoTestPath):
    train_path = trainPath
    test_path = testPath
    
    train_df = pd.read_csv( annoTrainPath , header = None,names = ['file','x_min','y_min','x_max','y_max','class'] )
    test_df = pd.read_csv( annoTestPath ,header = None,names = ['file','x_min','y_min','x_max','y_max','class'] )
    
    
    # Train images with its classification
    classification=[]
    #category = os.listdir(train_path)
    category = eda.get_immediate_subdirectories(train_path)
    for category in category:
        for i in range(0,len(os.listdir(os.path.join(train_path,category)))):
            image = os.listdir(os.path.join(train_path,category))[i]
            classification.append([image,category])
            
    #converting list to dataframe
    train_classification = pd.DataFrame(classification,columns=['file','classification'])
    
    
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
    train_df_final = train_df.merge(train_classification, on="file", how = 'inner')
    
    #adding file path in the dataframe
    train_df_final['file_path']=train_path+'/'+train_df_final['classification']+'/'+train_df_final['file']
    
    # Adding the height & width field with the dataframe
    train_df_final['width'] = 0
    train_df_final['height'] = 0
    
    # extracting the image width & height and loading to dataframe
    for i in range(0,train_df_final['file_path'].shape[0]):
        image = load_img(train_df_final['file_path'][i])
        train_df_final['width'][i]=image.size[0]
        train_df_final['height'][i]=image.size[1]
    
    
    test_df_final = test_df.merge(test_classification, on="file", how = 'inner')
    
    test_df_final['file_path']=test_path+'/'+test_df_final['classification']+'/'+test_df_final['file']
    
    # Adding the height & width field with the dataframe
    test_df_final['width'] = 0
    test_df_final['height'] = 0
    # extracting the image width & height and loading to dataframe
    for i in range(0,test_df_final['file_path'].shape[0]):
        image = load_img(test_df_final['file_path'][i])
        test_df_final['width'][i]=image.size[0]
        test_df_final['height'][i]=image.size[1]
        
        
    img_num = np.random.randint(0,train_df_final.shape[0])
    imgsize = 300
    img = cv2.imread(train_df_final.loc[img_num,'file_path'])
    img = cv2.resize(img,(imgsize,imgsize))
    w = train_df_final.loc[img_num,'width'].astype('int')
    h = train_df_final.loc[img_num,'height'].astype('int')
    xmi = int( train_df_final.loc[img_num,'x_min'])
    xma = int( train_df_final.loc[img_num,'x_max'])
    ymi = int( train_df_final.loc[img_num,'y_min'])
    yma = int( train_df_final.loc[img_num,'y_max'])
    
    xmi = int (xmi*imgsize/w)
    ymi = int( ymi*imgsize/h)
    xma = int( xma*imgsize/w)
    yma = int( yma*imgsize/h)
    cv2.rectangle(img, (xmi, ymi), (xma, yma),(0,255,0),2)
    
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    plt.suptitle(train_df_final.loc[img_num,'classification'])
    plt.imshow(img)
    #plt.show()
    st.image(img, use_column_width=True,clamp = True)
    
    
    img_size = 224
    
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=20,width_shift_range=0.3,height_shift_range=0.3,horizontal_flip=True,preprocessing_function=normalize_data)
    test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=normalize_data)
    
    train_generator = train_datagen.flow_from_dataframe(train_df_final,x_col='file_path',y_col='classification',target_size=(224,224),batch_size=32)
    test_generator = test_datagen.flow_from_dataframe(test_df_final,x_col='file_path',y_col='classification',target_size=(224,224),batch_size=32)
    
    model = tf.keras.applications.ResNet50(include_top=False,input_shape=(224,224,3),weights='imagenet')
    
    x1 = model.output
    x2 = tf.keras.layers.GlobalAveragePooling2D()(x1)
    x3 = tf.keras.layers.Dropout(0.5)(x2)
    x4 = tf.keras.layers.BatchNormalization()(x3)
    
    prediction = tf.keras.layers.Dense(196,activation='softmax')(x4)
    
    st.write('Number of layers in Model: ',len(model.layers))
    
    for layer in model.layers:
        layer.trainable = False
        
    final_model = tf.keras.models.Model(inputs=model.input,outputs=prediction)
    
    final_model.summary()
    
    final_model.compile(optimizer='adam',loss= 'categorical_crossentropy',metrics=['accuracy'])
    
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint('carclass_Res.h5',
                                                      save_best_only=True,
                                                      monitor='val_accuracy',
                                                      mode = 'max',
                                                      verbose=1)
    
    
    final_model.fit(train_generator,epochs=50,
                    steps_per_epoch=train_df_final.shape[0]//32,
                    validation_data=test_generator,
                    validation_steps=test_df_final.shape[0]//32,
                    callbacks=[model_checkpoint])
    


def normalize_data(img):
    return tf.keras.applications.resnet50.preprocess_input(img)
    
    