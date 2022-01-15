import cv2
import numpy as np
from time import perf_counter
import streamlit as st
from math import hypot
import math
import dlib
import time

st.title("Camera Filter")
run = st.checkbox("Run")

#SETUP CAMERA
frame_window = st.image([])
scan_option = st.selectbox("Option scan effect:", ('Normal','Left to Right','Top to Bottom', 'Right to Left', 'Bottom to Top', '2 ways Top and Bottom','Pig Nose','Noodle Dance'))
cap = cv2.VideoCapture(0)

#SETUP FOR FILTER

#for Time Warp
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
imgg = np.zeros((480, 640, 3), np.uint8)
i = 0
j = -1

#for Pig Nose
nose_image = cv2.imread("pig_nose.png")
nose_image = cv2.cvtColor(nose_image, cv2.COLOR_BGR2RGB)
nose_mask = np.zeros((480, 640), np.uint8)

#for Noodle Dance
fps = cap.get(cv2.CAP_PROP_FPS)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
scale_fact = 1
segment_count = fps*3
segment_height = int(height*scale_fact/segment_count)
frames = []
t1 = perf_counter()
#START
while run:
    _, new_frame = cap.read()
    new_frame = np.fliplr(new_frame)
    new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
    h, w = new_frame.shape[:2]

    if scan_option == 'Left to Right':           
        imgg[:, i+1: w, :] = new_frame[:, i+1: w, :]
        cv2.line(imgg, (i+1, 0), (i+1, h), (0, 255, 0), 2)
        imgg[:, i:i+1, :] = new_frame[:, i:i+1, :]      

        i += 1
        frame_window.image(imgg)    
    
    elif scan_option == '2 ways Top and Bottom':
        if i != h/2:
            imgg[i + 1:j+h, :, :] = new_frame[i + 1:j+h, :, :]
            imgg[-h+i:j-2, :, :] = new_frame[-h+i:j-2, :, :]
            cv2.line(imgg, (0, i + 1), (w, i + 1), (0, 255, 0), 2)
            cv2.line(imgg, (0, j - 2+h), (w, j - 2+h), (0, 255, 0), 2)
            imgg[i:i + 1, :, :] = new_frame[i:i + 1, :, :]
            imgg[j - 1:j, :, :] = new_frame[j - 1:j, :, :]

            i += 1
            j -= 1
            frame_window.image(imgg)
        else:
            i += h/2 
            j -= h/2
            frame_window.image(imgg)  

    elif scan_option == 'Top to Bottom':
        imgg[i + 1:h, :, :] = new_frame[i + 1:h, :, :]
        cv2.line(imgg, (0, i + 1), (w, i + 1), (0, 255, 0), 2)
        imgg[i:i + 1, :, :] = new_frame[i:i + 1, :, :]

        i += 1
        frame_window.image(imgg)

    elif scan_option == 'Right to Left':     
        imgg[:, -w:j-2, :] = new_frame[:, -w:j-2, :]
        cv2.line(imgg, (j-2+w, 0), (j-2+w, h), (0, 255, 0), 2)
        imgg[:, j-1:j, :] = new_frame[:, j-1:j, :]

        j -= 1
        frame_window.image(imgg) 

    elif scan_option == 'Bottom to Top':    
        imgg[-h:j-2, :, :] = new_frame[-h:j-2, :, :]
        cv2.line(imgg, (0, j - 2+h), (w, j - 2+h), (0, 255, 0), 2)
        imgg[j - 1:j, :, :] = new_frame[j - 1:j, :, :]

        j -= 1
        frame_window.image(imgg)    

    elif scan_option == 'Pig Nose':
        nose_mask.fill(0)
        gray_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)


        faces = detector(new_frame)

        for face in faces:
            #print(face)
            landmarks = predictor(gray_frame, face)

            # Nose coordinates
            top_nose = (landmarks.part(29).x, landmarks.part(29).y)
            center_nose = (landmarks.part(30).x, landmarks.part(30).y)
            left_nose = (landmarks.part(31).x, landmarks.part(31).y)
            right_nose = (landmarks.part(35).x, landmarks.part(35).y)

            nose_width = int(hypot(left_nose[0] - right_nose[0],
                                   left_nose[1] - right_nose[1]) * 1.7)
            nose_height = int(nose_width * 0.77)

            # New nose position
            top_left = (int(center_nose[0] - nose_width / 2),
                        int(center_nose[1] - nose_height / 2))
            bottom_right = (int(center_nose[0] + nose_width / 2),
                            int(center_nose[1] + nose_height / 2))

            # Adding the new nose
            nose_pig = cv2.resize(nose_image, (nose_width, nose_height))
            nose_pig_gray = cv2.cvtColor(nose_pig, cv2.COLOR_BGR2GRAY)
            _, nose_mask = cv2.threshold(nose_pig_gray, 25, 255, cv2.THRESH_BINARY_INV)

            nose_area = new_frame[top_left[1]: top_left[1] + nose_height,
                        top_left[0]: top_left[0] + nose_width]
            nose_area_no_nose = cv2.bitwise_and(nose_area, nose_area, mask=nose_mask)
            final_nose = cv2.add(nose_area_no_nose, nose_pig)

            new_frame[top_left[1]: top_left[1] + nose_height,
            top_left[0]: top_left[0] + nose_width] = final_nose
        imgg = new_frame
        frame_window.image(imgg)
    
    elif scan_option == 'Noodle Dance':
        if scale_fact != 1:
            new_frame = cv2.resize(new_frame,
                               (int(new_frame.shape[1]*scale_fact),
                                int(new_frame.shape[0]*scale_fact)))
    frames.append(new_frame)
    if len(frames) >= segment_count:    
        segments = []
        for i,frame in enumerate(frames):
            segments.append(frame[i*segment_height:(i+1)*segment_height])

        noodled_frame = np.concatenate(segments, axis=0)

        frames.pop(0)
        
        t2 = perf_counter()
        delay = int(1000/fps - (t2-t1)*1000)
        delay = delay if delay >1 else 1
        
        t1 = perf_counter()
        imgg = new_frame
        frame_window.image(imgg)
    else:
        _, new_frame = cap.read()
        new_frame = np.fliplr(new_frame)
        new_frame = cv2.cvtColor(new_frame, cv2.COLOR_BGR2RGB)
        img = new_frame 
        result = frame_window.image(img)
        if new_frame is None:
            break
    

else:
    st.write('Stopped')  


