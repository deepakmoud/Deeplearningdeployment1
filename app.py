from flask.helpers import send_file
from jinja2 import Template
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from os import path
import re
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
# Feature Scaling
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np
import matplotlib
import numpy as np
from flask import Flask, request, jsonify, render_template

import pandas as pd

# coding=utf-8
import sys
import os
import glob
import re
import cv2
from  PIL import Image, ImageOps
# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
import io


matplotlib.use('Agg')

app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

#------------------------------ Saving dataset---------------------------------
# this is the path to save dataset for preprocessing
pathfordataset = "static/data-preprocess/"
pathfordatasetNew = "data-preprocess/new/"  
 
app.config['DFPr'] = pathfordataset
app.config['DFPrNew'] = pathfordatasetNew
#------------------------------ Saving dataset for Linear regression-------------------------------------------
# this is the path to save dataset for single variable LR
pathforonevarLR = "static/Regression/onevarLR"
pathforonevarLRplot = "Regression/onevarLR/plot"
app.config['LR1VAR'] = pathforonevarLR
app.config['LR1VARplot'] = pathforonevarLRplot

#------------------------------ Saving image for K means-------------------------------------------
# this is the path to save figure of K menas
pathforelbowplot = "kmeans/plot"
#pathforonevarLRplot = "Regression/onevarLR/plot"
#app.config['LR1VAR'] = pathforonevarLR
app.config['elbowplot'] = pathforelbowplot
#print(app.config['elbowplot'])

# for index page
#------------------------------ Launcing undex page-------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')

#------------------------------Data Preprocessing-------------------------------------------
# for data preprocessing
def model_predict(file_path, model):
    img = image.load_img(file_path, target_size=(128, 128))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')
    preds = model.predict(x)
    return preds

@app.route('/downloadNewDataset')
def download_file():
    path1 = "static/data-preprocess/new/trained_dataset.csv"
    return send_file(path1,as_attachment=True)

#------------------------------Download Model-------------------------------------------
@app.route('/downloadmodel')
def download_model():
    path1 = "static/data-preprocess/model/model.pkl"
    return send_file(path1,as_attachment=True)

#------------------------------About us-------------------------------------------
@app.route('/aboutusnew')
def aboutusnew():
    return render_template('aboutusnew.html')
#------------------------------Artificial Neural network-------------------------------------------


@app.route('/ann')
def ann():
    return render_template('/ann/ann.html')

#------------------------------Signature Verificationn-------------------------------------------

def model_predict(file_path, model):
    img = image.load_img(file_path, target_size=(128, 128))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')
    preds = model.predict(x)
    return preds

@app.route('/ann/signatureverification/signatureverification')
def signatureverification():
    return render_template('/ann/signatureverification/signatureverification.html')


@app.route('/ann/signatureverification/signatureverification',  methods=['GET', 'POST'])
def signatureverification1():
   
    if request.method == 'POST':
        my_dataset = request.files['my_dataset']
        my_model_name = request.form['name_of_model']
        
        dataset_path = os.path.join(pathfordataset, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        print(my_dataset)
        get_dastaset = os.path.join(app.config['DFPr'],secure_filename(my_dataset.filename))
        print(get_dastaset)
        input=secure_filename(my_dataset.filename)
        extension= input.split(".")
        extension=extension[1]
        print(extension)
        model = load_model("static/data-preprocess/model/model_vgg19.h5")
        # Make prediction
        preds = model_predict(get_dastaset, model)

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        if(preds> 0.5):
            result= 'Genuine'
        elif(preds< 0.5):
            result='Forged'
        plt.plot(get_dastaset)

        
        fig = plt.gcf()
        img_name1 = 'vgg19'
        fig.savefig('static/kmeans/plot/vgg19.png', dpi=1500)
        #elbow_plot = os.path.join(app.config['elbowplot'], '%s.png' % img_name1)
        vgg19_plot = os.path.join('kmeans\plot', '%s.png' % img_name1)
        plt.clf()

        return render_template('/ann/signatureverification/signatureverificationoutput.html', model_name=my_model_name,my_dataset=my_dataset, pred=result, visualize=input )

#-----------------------Digit Recognition---------------------------------------------
model_digit = load_model("static/data-preprocess/model/MNISTANN.h5")

def import_and_predict(image_data):
  
  image_resized = cv2.resize(image_data, (28, 28)) 
   
  prediction = model_digit.predict(image_resized.reshape(1,784))
  print('Prediction Score:\n',prediction[0])
  thresholded = (prediction>0.5)*1
  print('\nThresholded Score:\n',thresholded[0])
  print('\nPredicted Digit:',np.where(thresholded == 1)[1][0])
  digit = np.where(thresholded == 1)[1][0]
  #st.image(image_data, use_column_width=True)
  return digit



@app.route('/ann/digit/digit')
def digit():
    return render_template('/ann/digit/digit.html')


@app.route('/ann/digit/digit',  methods=['GET', 'POST'])
def digit1():
   
    if request.method == 'POST':
        input_image = request.files['input_image']
        print(input_image)
        my_model_name = request.form['name_of_model']
        
        dataset_path = os.path.join(pathfordataset, secure_filename(input_image.filename))
        input_image.save(dataset_path)
        
        get_dastaset = os.path.join(app.config['DFPr'],secure_filename( input_image .filename))
        input=secure_filename(input_image.filename)
        
        image=Image.open(input_image)
        image=np.array(image)
        
        #image=np.array(input_image)
        preds = import_and_predict(image)

        

        return render_template('/ann/digit/digitoutput.html', model_name=my_model_name,my_dataset=input_image, pred=preds, visualize=input )

#----------------------Image Classification cat/ Dog------------------------------
model_cat = load_model("static/data-preprocess/model/FDPCNN1.h5")

def import_and_predict_cat(image_data):
  #x = cv2.resize(image_data, (48, 48)) 
  #img = image.load_img(image_data, target_size=(48, 48))
  #x = image.img_to_array(img)
  size=(64, 64)
  image=ImageOps.fit(image_data, size, Image.ANTIALIAS)
  img=np.asarray(image)
  img_reshape=np.expand_dims(img, axis=1)
  img_reshape=img[np.newaxis,...]
  result = model_cat .predict(img_reshape)
  print(result)
  #training_set.class_indices
  if result[0][0] == 1:
    prediction = "Dog" 
    
  else:
    prediction = 'Cat'
    #x = np.expand_dims(x, axis=1)
  
  
  return prediction


@app.route('/ann/cat/cat')
def cat():
    return render_template('/ann/cat/cat.html')


@app.route('/ann/cat/cat',  methods=['GET', 'POST'])
def cat1():
   
    if request.method == 'POST':
        input_image = request.files['input_image']
        print(input_image)
        my_model_name = request.form['name_of_model']
        
        dataset_path = os.path.join(pathfordataset, secure_filename(input_image.filename))
        input_image.save(dataset_path)
        
        get_dastaset = os.path.join(app.config['DFPr'],secure_filename( input_image .filename))
        input=secure_filename(input_image.filename)
        
        image=Image.open(input_image)
        #image=np.array(image)
        
        #image=np.array(input_image)
        preds = import_and_predict_cat(image)

        

        return render_template('/ann/cat/catoutput.html', model_name=my_model_name,my_dataset=input_image, pred=preds, visualize=input )


#-------------Signature recognition-----------------------------------------------
model_signaturerecognition = load_model("static/data-preprocess/model/signatureRecognition_VGG16folder_model.h5")
SIGNATURE_CLASSES = ['001', '002', '003','004','006','009','012','013','014','015','016','017','018','019','020','021','022','023','024','025','026','027','028','029','030','031','032','033','034','035','036','037','038','039','040','041','042','043','044','045','046','047','048','049','050','051','052','053','054','055','056','057','058','059','060','061','062','063','064','065','066','067','068','069']
def import_and_predict_recognition(image_data, model):
  #img = image.load_img(image_data, target_size=(224, 224))
  #image = image.img_to_array(img)
  #img_reshap= np.expand_dims(image, axis=0)
  #img_reshap = preprocess_input(img_reshap)
  size=(224, 224)
  image=ImageOps.fit(image_data, size, Image.ANTIALIAS)
  img=np.asarray(image)
  img_reshape=np.expand_dims(img, axis=1)
  img_reshape=img[np.newaxis,...]
  block4_pool_features = model.predict(img_reshape)
  label_index=block4_pool_features.argmax()
  print(block4_pool_features)
  result=SIGNATURE_CLASSES[label_index]
  return result


@app.route('/ann/signaturerecognition/signaturerecognition')
def signaturerecognition():
    return render_template('/ann/signaturerecognition/signaturerecognition.html')


@app.route('/ann/signaturerecognition/signaturerecognition',  methods=['GET', 'POST'])
def signaturerecognition1():
   
    if request.method == 'POST':
        input_image = request.files['input_image']
        
        my_model_name = request.form['name_of_model']
        
        dataset_path = os.path.join(pathfordataset, secure_filename(input_image.filename))
        input_image.save(dataset_path)
        
        get_dastaset = os.path.join(app.config['DFPr'],secure_filename( input_image .filename))
        input=secure_filename(input_image.filename)
        
        image=Image.open(input_image)
        #image=np.array(image)
        
        #image=np.array(input_image)
        preds = import_and_predict_recognition(image, model_signaturerecognition)

        

        return render_template('/ann/signaturerecognition/signaturerecognitionoutput.html', model_name=my_model_name,my_dataset=input_image, pred=preds, visualize=input )

#--------------------Animal Breed identification---------------------------------

model_breed = load_model("static/data-preprocess/model/resnet_model.h5")
def model_predict_breed(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds

@app.route('/ann/breed/breed')
def breed():
    return render_template('/ann/breed/breed.html')


@app.route('/ann/breed/breed',  methods=['GET', 'POST'])
def breed1():
   
    if request.method == 'POST':
        my_dataset = request.files['input_image']
        my_model_name = request.form['name_of_model']
        
        dataset_path = os.path.join(pathfordataset, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        
        get_dastaset = os.path.join(app.config['DFPr'],secure_filename(my_dataset.filename))
        
        input=secure_filename(my_dataset.filename)
        
        
        # Make prediction
        preds = model_predict_breed(get_dastaset, model_breed)

        pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(pred_class[0][0][1])               # Convert to string

        return render_template('/ann/breed/breedoutput.html', model_name=my_model_name,my_dataset=my_dataset, pred=result, visualize=input )
#-----------------------Character Recognition---------------------------------------------
model_char = load_model("static/data-preprocess/model/alphabet.h5")

def predict_char(image_data):
  
  test_image = image.load_img(image_data, target_size = (32,32))
  test_image = image.img_to_array(test_image)
  test_image = np.expand_dims(test_image, axis = 0)
  result = model_char.predict(test_image)
  result = get_result(result)
  return result
  
def get_result(result):
    if result[0][0] == 1:
        return('a')
    elif result[0][1] == 1:
        return ('b')
    elif result[0][2] == 1:
        return ('c')
    elif result[0][3] == 1:
        return ('d')
    elif result[0][4] == 1:
        return ('e')
    elif result[0][5] == 1:
        return ('f')
    elif result[0][6] == 1:
        return ('g')
    elif result[0][7] == 1:
        return ('h')
    elif result[0][8] == 1:
        return ('i')
    elif result[0][9] == 1:
        return ('j')
    elif result[0][10] == 1:
        return ('k')
    elif result[0][11] == 1:
        return ('l')
    elif result[0][12] == 1:
        return ('m')
    elif result[0][13] == 1:
        return ('n')
    elif result[0][14] == 1:
        return ('o')
    elif result[0][15] == 1:
        return ('p')
    elif result[0][16] == 1:
        return ('q')
    elif result[0][17] == 1:
        return ('r')
    elif result[0][18] == 1:
        return ('s')
    elif result[0][19] == 1:
        return ('t')
    elif result[0][20] == 1:
        return ('u')
    elif result[0][21] == 1:
        return ('v')
    elif result[0][22] == 1:
        return ('w')
    elif result[0][23] == 1:
        return ('x')
    elif result[0][24] == 1:
        return ('y')
    elif result[0][25] == 1:
        return ('z')

@app.route('/ann/character/character')
def character():
    return render_template('/ann/character/character.html')


@app.route('/ann/character/character',  methods=['GET', 'POST'])
def character1():
   
    if request.method == 'POST':
        my_dataset = request.files['input_image']
        my_model_name = request.form['name_of_model']
        
        dataset_path = os.path.join(pathfordataset, secure_filename(my_dataset.filename))
        my_dataset.save(dataset_path)
        
        get_dastaset = os.path.join(app.config['DFPr'],secure_filename(my_dataset.filename))
        
        input=secure_filename(my_dataset.filename)
        
        
        # Make prediction
        preds = predict_char(get_dastaset)

        

        return render_template('/ann/character/characteroutput.html', model_name=my_model_name,my_dataset=my_dataset, pred=preds, visualize=input )
    
#------------------------------Convolution  Neural network-------------------------------------------


@app.route('/cnn')
def cnn():
    return render_template('/cnn/cnn.html')

#------------------------------Face Recognition-------------------------------------------
model_face = load_model("static/data-preprocess/model/Facemodel.h5")
FACE_CLASSES = ['ben_afflek', 'elton_john','jerry_seinfeld','madonna','mindy_kaling']
def predict_face(image_data):
  #x = cv2.resize(image_data, (48, 48)) 
  #img = image.load_img(image_data, target_size=(48, 48))
  #x = image.img_to_array(img)
  size=(224, 224)
  image=ImageOps.fit(image_data, size, Image.ANTIALIAS)
  img=np.asarray(image)
  img_reshape=np.expand_dims(img, axis=1)
  img_reshape=img[np.newaxis,...]
  features = model_face.predict(img_reshape)
  
  label_index=features.argmax()
  print(label_index)
  
  
  
  return FACE_CLASSES[label_index]

@app.route('/cnn/face/face')
def face():
    return render_template('/cnn/face/face.html')


@app.route('/cnn/face/face',  methods=['GET', 'POST'])
def face1():
   
    if request.method == 'POST':
        input_image = request.files['input_image']
        
        my_model_name = request.form['name_of_model']
        
        dataset_path = os.path.join(pathfordataset, secure_filename(input_image.filename))
        input_image.save(dataset_path)
        
        get_dastaset = os.path.join(app.config['DFPr'],secure_filename( input_image .filename))
        input=secure_filename(input_image.filename)
        
        image=Image.open(input_image)
        #image=np.array(image)
        
        #image=np.array(input_image)
        preds = predict_face(image)

        

        return render_template('/cnn/face/faceoutput.html', model_name=my_model_name,my_dataset=input_image, pred=preds, visualize=input )

#------------------------------Face Expression Recognition-------------------------------------------
model_faceexpression = load_model("static/data-preprocess/model/FaceExpressionmodel.h5")
FACE_CLASSES1 = ['angry', 'disgust','fear','happy','neutral', 'sad', 'surprise']
def predict_faceexpression(image_data):
  size=(48, 48)
  image=ImageOps.fit(image_data, size, Image.ANTIALIAS)
  img=np.asarray(image)
  img_reshape=np.expand_dims(img, axis=1)
  img_reshape=img[np.newaxis,...]
  features = model_faceexpression.predict(img_reshape)
  #x = np.expand_dims(x, axis=1)
  #x = preprocess_input(x)
  #features = model.predict(x)
  
  label_index=features.argmax()
  
  
  return FACE_CLASSES1[label_index]

@app.route('/cnn/faceexpression/faceexpression')
def faceexpression():
    return render_template('/cnn/faceexpression/faceexpression.html')


@app.route('/cnn/faceexpression/faceexpression',  methods=['GET', 'POST'])
def faceexpression1():
   
    if request.method == 'POST':
        input_image = request.files['input_image']
        
        my_model_name = request.form['name_of_model']
        
        dataset_path = os.path.join(pathfordataset, secure_filename(input_image.filename))
        input_image.save(dataset_path)
        
        get_dastaset = os.path.join(app.config['DFPr'],secure_filename( input_image .filename))
        input=secure_filename(input_image.filename)
        
        image=Image.open(input_image)
        #image=np.array(image)
        
        #image=np.array(input_image)
        preds = predict_faceexpression(image)

        

        return render_template('/cnn/faceexpression/faceexpressionoutput.html', model_name=my_model_name,my_dataset=input_image, pred=preds, visualize=input )
#----------------Object detection----------------------
import math
from cv2 import *
import os
directory = r'F:\ML\Deep Learning Lab Deployment\static\data-preprocess\new'
cascade_face = cv2.CascadeClassifier('static/data-preprocess/model/haarcascade_frontalface_default.xml') 
cascade_eye = cv2.CascadeClassifier('static/data-preprocess/model/haarcascade_eye.xml') 
cascade_smile = cv2.CascadeClassifier('static/data-preprocess/model/haarcascade_smile.xml')
def detection(grayscale, img):
    face = cascade_face.detectMultiScale(grayscale, 1.3, 5)
    for (x_face, y_face, w_face, h_face) in face:
        cv2.rectangle(img, (x_face, y_face), (x_face+w_face, y_face+h_face), (255, 130, 0), 2)
        ri_grayscale = grayscale[y_face:y_face+h_face, x_face:x_face+w_face]
        ri_color = img[y_face:y_face+h_face, x_face:x_face+w_face] 
        eye = cascade_eye.detectMultiScale(ri_grayscale, 1.2, 18) 
        for (x_eye, y_eye, w_eye, h_eye) in eye:
            cv2.rectangle(ri_color,(x_eye, y_eye),(x_eye+w_eye, y_eye+h_eye), (0, 180, 60), 2) 
        smile = cascade_smile.detectMultiScale(ri_grayscale, 1.7, 20)
        for (x_smile, y_smile, w_smile, h_smile) in smile: 
            cv2.rectangle(ri_color,(x_smile, y_smile),(x_smile+w_smile, y_smile+h_smile), (255, 0, 130), 2)
    return img 
@app.route('/object')
def object():
    return render_template('/object/object.html')

@app.route('/object/smile/smile')
def smile():
    return render_template('/object/smile/smile.html')

@app.route('/object/smile/smile',  methods=['GET', 'POST'])
def smiledetection():
    if request.method == 'POST':
        input_image = request.files['input_image']
    
        key = cv2. waitKey(1)
        webcam = cv2.VideoCapture(0) 

    while True:
     try:
        check, frame = webcam.read()
        print(check) #prints true as long as the webcam is running
        print(frame) #prints matrix values of each framecd 
        #cv2.imshow("Capturing", frame)
        key = cv2.waitKey(1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        canvas = detection(gray, frame)
        cv2.imshow('Video is Running. Press S on Keyboard to save Image', canvas)
        if key == ord('s'): 
            os.chdir(directory)
            cv2.imwrite(filename='saved_img.jpg',img=frame)
            imageoutput = os.path.join(app.config['DFPrNew'], 'saved_img.jpg')
                     
            #filename ='F:\ML\Deep Learning Lab Deployment\static\data-preprocess\new\saved_img.jpg'
            webcam.release()
            cv2.waitKey(1650)
            cv2.destroyAllWindows()
           # print("Processing image...")
            #img_ = cv2.imread('saved_img.jpg', cv2.IMREAD_ANYCOLOR)
            #print("Converting RGB image to grayscale...")
            #gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
            #print("Converted RGB image to grayscale...")
            #print("Resizing image to 28x28 scale...")
            #img_ = cv2.resize(gray,(28,28))
            #print("Resized...")
            #img_resized = cv2.imwrite(filename='saved_img-final.jpg', img=img_)
            #print("Image saved!")        
            break
        elif key == ord('q'):
            print("Turning off camera.")
            imageoutput = os.path.join(app.config['DFPrNew'], 'Deepak Moud.jpg')
            #imageoutput = os.path.join(app.config['DFPr'],secure_filename( input_image .filename))
            #dataset_path = os.path.join(pathfordatasetNew, secure_filename(input_image.filename))
            #input_image.save(dataset_path)
            #dataset_path = os.path.join(pathfordataset, secure_filename(input_image.filename))
            #print(dataset_path)
            #input_image.save(dataset_path)
        
            #imageoutput = os.path.join(app.config['DFPr'],secure_filename(input_image.filename))
            #print(imageoutput)
            #imageoutput=secure_filename(input_image.filename)
            #imageoutput = os.path.join(app.config['DFPr'],secure_filename(input_image.filename))
            #input_image.save(imageoutput)
            #imageoutput=secure_filename(input_image.filename)
            #webcam.release()
            #print("Camera off.")
            #print("Program ended.")
            cv2.destroyAllWindows()
            break
         
     except(KeyboardInterrupt):
            print("Turning off camera.")
            webcam.release()
            print("Camera off.")
            print("Program ended.")
            cv2.destroyAllWindows()
            break
    #while True:
     #   frameId = vc.get(1) # current frame number
      #  print(frameId)
       # frameRate = vc.get(5) # frame rate
        #print(frameRate)
        #ret, frame = vc.read() # We get the last frame.
        #namedWindow("cam-test")
        #imshow("cam-test",frame)
        #filename = imagesFolder + "/image_" +  str(int(frameId)) + ".jpg"
        #print(filename)
        #imwrite(filename,frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
         #   break 
        
         #save image
        #filename = imagesFolder + "/image_" +  str(int(frameId)) + ".jpg"
        #imwrite(filename,frame) #save image
        #if cv2.waitKey(1) & 0xFF == ord('q'):
           #break
        #destroyWindow("cam-test")1
        
        #if (ret != True):
            #break
        #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # We do some colour transformations.
        
        #canvas = detection(gray, frame) # We get the output of our detect function.ret, frame = cap.read()
        
        #if (frameId % math.floor(frameRate) == 0):
            
            
            #filename = imagesFolder + "/image_" +  str(int(frameId)) + ".jpg"
            #print(filename)
            #cv2.imwrite(filename, frame)
            #print(cv2.imwrite(filename, frame))
            
            #_, img = vc.read() 
        #grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        #final = detection(grayscale, img) 
        #cv2.imshow('Video', canvas)
        #filename = imagesFolder + "/image_" + str(int(frameId)) + ".jpg"
        #cv2.imwrite(filename, frame)
       # if cv2.waitKey(1) & 0xFF == ord('q'):
           # break 
        


            
    #vc.release()

   
   
     
    return render_template('/object/smile/smileoutput.html', model_name="Object Detection", visualize=imageoutput)
    
  #------------------------------Recurrent Neural Network  -------------------------------------------


@app.route('/rnn')
def rnn():
    return render_template('/rnn/rnn.html')

#------------------------------Face Recognition-------------------------------------------
model_moviereview = load_model("static/data-preprocess/model/nlpmovierreview.h5")
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
tokenizer = Tokenizer(num_words=1000)
def predict_moview_review(text):
  pad='pre'
  tokens = tokenizer.texts_to_sequences(text)
  tokens_pad = pad_sequences(tokens, maxlen=544,padding=pad, truncating=pad)
  result= model_moviereview.predict(tokens_pad)
  result=result[0][0]
  if result > 0.5:
    result="Positive"
  else:
    result="Negative"
  
  return result
@app.route('/rnn/moviereview/moviereview')
def moviereview():
    return render_template('/rnn/moviereview/moviereview.html')


@app.route('/rnn/moviereview/moviereview',  methods=['GET', 'POST'])
def moviewreview1():
   
    if request.method == 'POST':
        review_input = request.form['review']
        
       
        
        
        pred=predict_moview_review(review_input)
        

        return render_template('/rnn/moviereview/moviereviewoutput.html', pred=pred)  

#-------------------Flask Application--------------------------------------------
if __name__ == '__main__':
    app.run(debug=True)
    
# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response
app.config["CACHE_TYPE"] = "null"




