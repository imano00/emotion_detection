import tensorflow 
from keras.models import load_model
from time import sleep
# from keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
import random
from itertools import count
import time
from matplotlib.animation import FuncAnimation
from mpl_toolkits import mplot3d
import mysql.connector
from datetime import datetime


mydb = mysql.connector.connect(
  host="localhost",
  user="root",
  password="",
  database="emotiondetection",
)

#face_classifier = cv2.CascadeClassifier(r'C:\Users\Admin\Desktop\PythonProject\EmotionDetectionCNN\haarcascade_frontalface_default.xml')
#classifier =load_model(r'C:\Users\Admin\Desktop\PythonProject\EmotionDetectionCNN\model.h5')
face_classifier = cv2.CascadeClassifier(r'C:\Users\imon\OneDrive - Universiti Tenaga Nasional\Desktop\Emotion_Detection_CNN-main\Experiment 1\haarcascade_frontalface_default.xml')
classifier =load_model(r'C:\Users\imon\OneDrive - Universiti Tenaga Nasional\Desktop\Emotion_Detection_CNN-main\Experiment 1\model.h5')

emotion_labels = ['Angry','Disgust','Fear','Happy','Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('http://192.168.1.107:4747/video')
# cap = cv2.VideoCapture('http://172.17.105.169:4747/video')  

# conn = sqlite3.connect("test.db")

# cursorObject = conn.cursor()

# # create a table
# cursorObject.execute("CREATE TABLE emotiondata(id string, img blob)")
# conn.commit()

# im = open( 'gfg.png', 'rb' ).read()


while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x,y,w,h) in faces:

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        roi_gray = gray[y:y+h,x:x+w]
        roi_gray = cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray])!=0:
            roi = roi_gray.astype('float')/255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi,axis=0)

            # prediction = classifier.predict(roi)[0]
            # label=emotion_labels[prediction.argmax()]
            # label_position = (x,y)
            # cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)

            prediction = classifier.predict(roi)[0]
            label=emotion_labels[prediction.argmax()]
            label_position = (x,y)            

            prediction_str = str(prediction)
            prediction_str_position = (x,y) 
            # print(type(prediction))

            font=(cv2.FONT_HERSHEY_DUPLEX)

            # testing panda

            array_test = np.array([prediction])

            df = pd.DataFrame(array_test, columns = ['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise'])

            dfr = df.round(3)
            print(dfr)
            # df.plot.bar()

            fig, ax = plt.subplots(dpi=240)
            area = pd.DataFrame(df, columns=['Angry','Disgust','Fear','Happy','Neutral','Sad','Surprise'])
            # area.plot(kind='area',ax=ax,stacked=False)

            # plt.title('Demo graph for Area plot')
            # plt.show()

            # def update_line

            happy=dfr['Happy']
            angry=dfr['Angry']
            disgust=dfr['Disgust']
            fear=dfr['Fear']
            neutral=dfr['Neutral']
            surprise=dfr['Surprise']
            sad=dfr['Sad']

            dfs = df.to_string()

            # happy=dfs['Happy']
            # angry=dfs['Angry']
            # disgust=dfs['Disgust']
            # fear=dfs['Fear']
            # neutral=dfs['Neutral']
            # surprise=dfs['Surprise']
            # sad=dfs['Sad']

            happys = happy.to_string()
            angrys = angry.to_string()
            disgusts = disgust.to_string()
            fears = fear.to_string()
            neutrals = neutral.to_string()
            surprises = surprise.to_string()
            sads = sad.to_string()

            # print(type(dfs))
            # y = df.get('Happy')
            # z = df.get('Angry')
            # q = df.get('Disgust')
            # a = df.get('Fear')
            # b = df.get('Neutral')
            # c = df.get('Suprise')
            # d = df.get('Sad')
        
            # print(type(neutrals))

            # print(b)
            # print(df)
            
            # print(type(df))
            # print(plt.style.available)

            # test graph animate

            # plt.style.use('seaborn-bright')
            # x_values = []
            # y_values = []
            # z_values = []
            # q_values = []
            # a_values = []
            # b_values = []
            # c_values = []
            # d_values = []

            # counter = 0
            
            # index = count()

            # for x in df:

            #     def animate(i):

            #         #print(counter)
                    

            #         x = next(index) # counter or x variable -> index
            #         counter = next(index)
            #         # print(counter)
            #         x_values.append(x)

            #         '''
            #         Three random value series ->
            #         Y : 0-5
            #         Z : 3-8
            #         Q : 0-10
            #         '''

            #         happy=df['Happy']
            #         angry=df['Angry']
            #         disgust=df['Disgust']
            #         fear=df['Fear']
            #         neutral=df['Neutral']
            #         surprise=df['Surprise']
            #         sad=df['Sad']
        
            #         y = happy
            #         z = angry
            #         q = disgust
            #         a = fear
            #         b = neutral
            #         c = surprise
            #         d = sad
                    
            #         # print(y)
            #         print(y,z,q,a,b,c,d)
                    
            #         # append values to keep graph dynamic
            #         # this can be replaced with reading values from a csv files also
            #         # or reading values from a pandas dataframe
            #         y_values.append(y)
            #         z_values.append(z)
            #         q_values.append(q)
            #         a_values.append(a)
            #         b_values.append(b)
            #         c_values.append(c)
            #         d_values.append(d)
                    
            #         print(y_values,z_values,q_values,a_values,b_values,c_values,d_values)

            #         if counter >40:
            #             '''
            #             This helps in keeping the graph fresh and refreshes values after every 40 timesteps
            #             '''
            #             x_values.pop(0)
            #             y_values.pop(0)
            #             z_values.pop(0)
            #             q_values.pop(0)
            #             a_values.pop(0)
            #             b_values.pop(0)
            #             c_values.pop(0)
            #             d_values.pop(0)                    
            #             #counter = 0
            #             plt.cla() # clears the values of the graph

            #         # plt.plot(x_values, x_values,linestyle='--')   
            #         plt.plot(x_values, y_values,linestyle='--')
            #         plt.plot(x_values, z_values,linestyle='--')
            #         plt.plot(x_values, q_values,linestyle='--')
            #         plt.plot(x_values, a_values,linestyle='--')
            #         plt.plot(x_values, b_values,linestyle='--')
            #         plt.plot(x_values, c_values,linestyle='--')        
            #         plt.plot(x_values, d_values,linestyle='--')    
                    
            #         ax.legend(["Happy","Angry","Disgust","Fear","Neutral","Suprise","Sad"])
            #         ax.set_xlabel("X values")
            #         ax.set_ylabel("Values for 7 Different Emotions")
            #         plt.title('Dynamic line graphs')
                    
            #         time.sleep(.25) # keep refresh rate of 0.25 seconds

            #     ani = FuncAnimation(plt.gcf(), animate, 1000)            
            #     # plt.tight_layout()
            #     # plt.show()    

            #     # if plt.waitKey(1) & 0xFF == ord('q'):
            #         # break    


            if label == 'Angry':
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_DUPLEX,1,(0,0,255),2)                        
            elif label == 'Disgust':
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,128,0),2)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_DUPLEX,1,(0,128,0),2)    
            elif label == 'Fear':
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,43,159),2)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_DUPLEX,1,(255,43,159),2)                
            elif label == 'Happy':
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,165,0),2)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_DUPLEX,1,(255,165,0),2)                
            elif label == 'Neutral':
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,255,255),2)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)                
                # cv2.putText(frame,neutrals,prediction_str_position,cv2.FONT_HERSHEY_DUPLEX,1,(255,255,255),2)              
            elif label == 'Sad':
                cv2.rectangle(frame,(x,y),(x+w,y+h),(128,128,128),2)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_DUPLEX,1,(128,128,128),2)   
            else:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(234,221,202),2)
                cv2.putText(frame,label,label_position,cv2.FONT_HERSHEY_DUPLEX,1,(234,221,202),2)                

        else:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(128,128,128),2)
            cv2.putText(frame,'No Faces',(30,80),cv2.FONT_HERSHEY_DUPLEX,1,(0,255,0),2)

    cv2.imshow('Emotion Detection System',frame)

    # k = cv2.waitKey(wait_key) & 0xff
    # if chr(k) == 'r':  #start running
    #     wait_key = 30
    # elif chr(k) == 'p':  #pause between frames
    #     wait_key = 0
    # elif k == 27:  #end processing
    #     break
    # else:
    #     k = 0

    # if plt.waitKey(2) & 0xFF == ord('w'):
        # break

    if cv2.waitKey(1) & 0xFF == ord('w'):
        # plt.savefig("graph.jpg")
        name = input('Enter your name?\n')     # \n ---> newline  ---> It causes a line break
        print(name)

        # Getting the current date and time
        dt = datetime.now()

        mycursor = mydb.cursor()

        sql = "INSERT INTO analysis (name, emotion, datetime) VALUES (%s, %s, %s)"
        val = (name, label, dt)
        mycursor.execute(sql, val)

        mydb.commit()

        print(mycursor.rowcount, "Record inserted.")
        

    elif cv2.waitKey(1) & 0xFF == ord('e'):
        cv2.destroyAllWindows()
        break

        

cap.release()
cv2.destroyAllWindows()