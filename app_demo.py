# -*- coding: utf-8 -*-
"""
Created on Sat May 29 16:07:53 2021

@author: Manoj-PC
"""

import streamlit as st
from PIL import Image
import cv2 
import numpy as np
import pandas as pd


import tkinter as tk
from tkinter import filedialog
from caps_utility import eda_visualization as eda
from caps_utility import FRCNN as frcnn
from caps_utility import testFRCNN as testFRCNN
from caps_utility import predictFrcnn as predictbox
from caps_utility import CarClassification_G as classResnetG
from caps_utility import predictClassG as predictClass

from caps_utility import SessionState
#import naming as n

    
trainPath = ''
testpath = ''
annoTrainPath = ''
annoTestPath = ''

df_train = pd.DataFrame()
df_test = pd.DataFrame()

test_imgs = []
classes_count = []
class_mapping = []

def main():
      
    
    selected_box = st.sidebar.selectbox(
        'Choose one of the following',
        ('Welcome', 'EDA', 'Model Training - Object Detection','Model Training - Object Classification', 'Prediction - Object Detection', 'Prediction - Object Classification','Predict')
        )

    
    if selected_box == 'Welcome':
        welcome() 
    if selected_box == 'EDA':
        dataViz()
    if selected_box == 'Model Training - Object Detection':
        modelTrainBbox()
    if selected_box == 'Prediction - Object Detection':
        modelPredictBbox()
    if selected_box == 'Model Training - Object Classification':
        modelTrainClassG()
    if selected_box == 'Prediction - Object Classification':
        modelPredictClassG()
    if selected_box == 'Test Set Prediction - Object Classification':
        modelTestsetPredictClassG()
    if selected_box == 'Test Set Prediction - Object Detection':
        modelTestsetPredictBbox()
    if selected_box == 'Predict':
        predictAll()


def welcome():
    
    st.title('Car Classification and Detection')

    st.subheader('A app that helps is classifiying and detecting cars in a image. Shows different steps involved in the process and use it independently.'
                 + ' You can choose the options'
                 + ' from the left.')
    
    st.image(['sampleImages\\1.png' , 'sampleImages\\2.png', 'sampleImages\\3.png', 'sampleImages\\4.png', 'sampleImages\\5.png', 'sampleImages\\6.png'],width=200) #,use_column_width=True)
    
    st.subheader('Jun20A Group 6B')
    st.text('-Antara')
    st.text('-Deepiga')
    st.text('-Gowtham')
    st.text('-Manoj')
    st.text('-Shashank')
    #,sampleImages\\2.png,sampleImages\\3.png,sampleImages\\4.png,sampleImages\\5.png,sampleImages\\6.png


def dataViz():
      
    # Set up tkinter
    # root = tk.Tk()
    # root.withdraw()
    

    
    # # Make folder picker dialog appear on top of other windows
    # root.wm_attributes('-topmost', 1)
     
    st.title('Exploratory Data Analysis')
    
    

    # for maintaining session for multiple button or inputs in same page
    # session_state = SessionState.get(name="", button_sent=False)
    # button_sent = st.button("SUBMIT")
    # if button_sent or session_state.button_sent: # <-- first time is button interaction, next time use state to go to multiselect
    #     session_state.button_sent = True
    #     listnames = n.show_names(name)
    #     selectednames=st.multiselect('Select your names',listnames)
    #     st.write(selectednames)
    
    
    st.write('Please select train folder:')

    
    trainPath = st.text_input('Train Folder:','E:\GreatLearning\Capstone_Project\CapstoneProject_GL\data\Car Images-20210501T094840Z-001\Car Images\Train Images')
    testPath = st.text_input('Test Folder:','E:\GreatLearning\Capstone_Project\CapstoneProject_GL\data\Car Images-20210501T094840Z-001\Car Images\Test Images')
    annoTrainPath = st.text_input('Train annotation file:' , 'E:\GreatLearning\Capstone_Project\CapstoneProject_GL\data\Annotations-20210510T185520Z-001\Annotations\Train Annotations.csv')
    annoTestPath = st.text_input('Test annotation file:', 'E:\GreatLearning\Capstone_Project\CapstoneProject_GL\data\Annotations-20210510T185520Z-001\Annotations\Test Annotation.csv')
    
    clicked1 = st.sidebar.button('Start EDA')
    if clicked1:
        #st.write('Train Selected folder:',trainPath)
        #st.write('Test Selected folder:',testPath)
        #st.write('Train Selected annotation file:',annoTrainPath)
        #st.write('Test Selected annotation file:',annoTestPath)
        eda.process(trainPath, testPath, annoTrainPath, annoTestPath)


def modelTrainBbox():
    st.title('Train Model for object detection')
    #frcnn.demo()
    st.markdown('## Train & Test folder details')    
    trainPath = st.text_input('Train Folder:','E:\GreatLearning\Capstone_Project\CapstoneProject_GL\data\Car Images-20210501T094840Z-001\Car Images\Train Images')
    trainImgCount = st.text_input('Number of images to pick from train folders:','20')
    testPath = st.text_input('Test Folder:','E:\GreatLearning\Capstone_Project\CapstoneProject_GL\data\Car Images-20210501T094840Z-001\Car Images\Test Images')
    testImgCount = st.text_input('Number of images to pick from test folders:','2')
    annoTrainPath = st.text_input('Train annotation file:' , 'E:\GreatLearning\Capstone_Project\CapstoneProject_GL\data\Annotations-20210510T185520Z-001\Annotations\Train Annotations.csv')
    annoTestPath = st.text_input('Test annotation file:', 'E:\GreatLearning\Capstone_Project\CapstoneProject_GL\data\Annotations-20210510T185520Z-001\Annotations\Test Annotation.csv')
    
    st.markdown('## Image Augmentation - Parameters')
    horiFlag = st.selectbox("Horizontal Flip of images: ", ('False','True'))
    vertFlag = st.selectbox("Vertical Flip of images: ", ('False','True'))
    rot90Flag = st.selectbox("Rotate 90 for train images: ", ('False','True'))
    
    st.markdown('## Model training parameters')
    num_epochs = st.text_input('Number of epochs:', '2')
    
    
    
    
    
    configFile = 'config_ui.pickle'
    
    clicked1 = st.sidebar.button('Start Training')
    if clicked1:
        frcnn.getAnnotatedData(trainPath, testPath, annoTrainPath, annoTestPath,int(trainImgCount),int(testImgCount))
        #clicked2 = st.button('Start Training')
        #if clicked2:
        #    frcnn.trainFRRCNN()
        frcnn.trainFRRCNN(horiFlag, vertFlag, rot90Flag, configFile, int(num_epochs))
        
    clicked2 = st.sidebar.button('Start Validation with Test data set')
    if clicked2:
        testFRCNN.testModel(configFile)

def modelPredictBbox():
    st.title('Predict using Model for object detection')
    config_list = st.sidebar.selectbox("Select Config File", ('Default','Newly Trained'))
    # if config_list is not None:
    #     st.write('You selected:', config_list , ' config file')
    # else:
    #     config_list='Default'
    #     st.write('You selected:', config_list, ' config file')
    
    st.write('You selected:', config_list, ' config file')
    
    uploaded_file = st.sidebar.file_uploader("Choose a image",accept_multiple_files = True)
    if uploaded_file is not None:
        #for i, img in enumerate(uploaded_file):
            #im = Image.open(img)
            #st.image(im, caption="Input Image: " + str(i+1) , width=100)
            #file_details = {"Filename":img.name,"FileType":img.type,"FileSize":img.size} #,"FileShape":img.shape}
            #st.write(file_details)
        st.markdown('## Selected Images preview:')
        st.image(uploaded_file, width=100)
            
        
    clicked1 = st.button('Predict Bounding Boxes')
    if clicked1 and (uploaded_file is not None):
        if config_list == 'Default':
            config_filename = 'configPostSubmission_FullData.pickle'
        else:
            config_filename = 'config_ui.pickle'
        predictbox.predictImg(uploaded_file, config_filename)
    # else:
    #     st.markdown('### Select the Options')

 
def modelTrainClassG():
    st.title('Train Model for object classification')
    #frcnn.demo()
        
    trainPath = st.text_input('Train Folder:','E:\GreatLearning\Capstone_Project\CapstoneProject_GL\data\Car Images-20210501T094840Z-001\Car Images\Train Images')
    #trainImgCount = st.text_input('Number of images to pick from train folders:','20')
    testPath = st.text_input('Test Folder:','E:\GreatLearning\Capstone_Project\CapstoneProject_GL\data\Car Images-20210501T094840Z-001\Car Images\Test Images')
    #testImgCount = st.text_input('Number of images to pick from test folders:','2')
    annoTrainPath = st.text_input('Train annotation file:' , 'E:\GreatLearning\Capstone_Project\CapstoneProject_GL\data\Annotations-20210510T185520Z-001\Annotations\Train Annotations.csv')
    annoTestPath = st.text_input('Test annotation file:', 'E:\GreatLearning\Capstone_Project\CapstoneProject_GL\data\Annotations-20210510T185520Z-001\Annotations\Test Annotation.csv')
    
    
    clicked1 = st.sidebar.button('Start Training Resnet')
    if clicked1:
        classResnetG.trainModel(trainPath, testPath, annoTrainPath, annoTestPath) 
        #,int(trainImgCount),int(testImgCount))
   

def modelPredictClassG():
    
    st.title('Predict car class')
    weightFiles = st.sidebar.selectbox("Select weight file", ('Default','Newly Trained'))
    st.write('You selected:', weightFiles, ' weights')

    uploaded_file = st.sidebar.file_uploader("Choose a image",accept_multiple_files = True)
    if uploaded_file is not None:
        #session_state.testButton = False
        st.markdown('## Selected Images preview:')
        st.image(uploaded_file, width=100)

        
    clicked1 = st.button('Predict car class')
    if clicked1 and (uploaded_file is not None):
        if weightFiles == 'Default':
            weights = '196carclass_Reswithcrop.h5'
        else:
            weights = 'carclass_Res.h5'
        #model = predictClass.loadWeights(weights)
        predictClass.predictImg(uploaded_file, weights)
    

def modelTestsetPredictClassG():
    st.title('Predict car class from test data set')
    weightFiles = st.sidebar.selectbox("Select weight file", ('Default','Newly Trained'))      
    session_state = SessionState.get(name="", testButton=False)
    #session_state = SessionState.get(name="", predictdf=False)
    testButton = st.sidebar.button('Predict from test data set')
    if testButton or session_state.testButton:
        #uploaded_file = None
        session_state.testButton = True
        testPath = st.text_input('Test Folder:','E:\GreatLearning\Capstone_Project\CapstoneProject_GL\data\Car Images-20210501T094840Z-001\Car Images\Test Images')
        annoTestPath = st.text_input('Test annotation file:', 'E:\GreatLearning\Capstone_Project\CapstoneProject_GL\data\Annotations-20210510T185520Z-001\Annotations\Test Annotation.csv')
        num = st.text_input('Number from test data set: ', '10')
        predictdf = st.button('Predicct test data')
        #if predictdf or session_state.predictdf:
        if predictdf:
            if weightFiles == 'Default':
                weights = '196carclass_Reswithcrop.h5'
            else:
                weights = 'carclass_Res.h5'
            #session_state2.predictdf = True
            
            predictClass.predictTestImg(testPath, annoTestPath, int(num), weights)
    


def predictAll():
    
    st.title('Predict car class & objection')
    weightFiles = st.sidebar.selectbox("Select weight file", ('Default','Newly Trained'))
    config_list = st.sidebar.selectbox("Select Config File", ('Default','Newly Trained'))
    st.write('You selected:', weightFiles, ' weights')
    st.write('You selected:', config_list, ' config file')
    
       
 
    uploaded_file = st.sidebar.file_uploader("Choose a image",accept_multiple_files = True)
    if uploaded_file is not None:

        st.markdown('## Selected Images preview:')
        st.image(uploaded_file, width=100)
            
        
    clicked1 = st.button('Predict')
    if clicked1 and (uploaded_file is not None):
        st.write('Car classification started.')
        if weightFiles == 'Default':
            weights = '196carclass_Reswithcrop.h5'
        else:
            weights = 'carclass_Res.h5'
        predictClass.predictImg(uploaded_file, weights, False)
        
        st.write('Car bounding box prediction started.')
        if config_list == 'Default':
            config_filename = 'configPostSubmission_FullData.pickle'
        else:
            config_filename = 'config_ui.pickle'
        predictbox.predictImg(uploaded_file, config_filename)
        


        
def modelTestsetPredictBbox():
    st.title('Predict car bounding box from test data set')
    config_list = st.sidebar.selectbox("Select Config File", ('Default','Newly Trained')) 
    if config_list == 'Default':
        config_filename = 'configPostSubmission_FullData.pickle'
    else:
        config_filename = 'config_ui.pickle'
    
    st.write('You selected:', config_list, ' config file')    
    
    testPath = st.text_input('Test Folder:','E:\GreatLearning\Capstone_Project\CapstoneProject_GL\data\Car Images-20210501T094840Z-001\Car Images\Test Images')
    annoTestPath = st.text_input('Test annotation file:', 'E:\GreatLearning\Capstone_Project\CapstoneProject_GL\data\Annotations-20210510T185520Z-001\Annotations\Test Annotation.csv')
    testImgCount = st.text_input('Number of images to pick from test folders:','2')
   # loadBtn = st.button('Load test set data')

    #if loadBtn:
    #    test_imgs, classes_count, class_mapping = testFRCNN.loadtestsetdata(testPath, annoTestPath, int(testImgCount), config_filename)
    #    st.write('Lenght of images :', len(test_imgs))
    
    num = st.text_input('Number from test data set: ', '10')
    
    predictdf = st.button('Predict from test data')
    
    if predictdf:
        predictbox.predictTestsetImg( testPath, annoTestPath, int(testImgCount), int(num), config_filename)

    
if __name__ == "__main__":
    main()