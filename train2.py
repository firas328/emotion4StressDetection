import mediapipe as mp 
import cv2
import numpy as np
import matplotlib.pyplot as plt
blinks=[]
blinkStress=[]
eyeDirectionStress=[]
eyeBrowCentroids=[]
eyeBrowStress=[]

mp_face_mesh=mp.solutions.face_mesh
cap = cv2.VideoCapture("video.mp4")
with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:
    while True:
        ret,frame=cap.read()
        if not ret:
          print("Ignoring empty  frame.")
        
          break
        results=face_mesh.process(frame)
        landmarks=results.multi_face_landmark[0]
        ###detect stress from eyebrows ###
        #we will get stress based on eyebrow motion
        # generally eye brow move vertically so we focus on y axe 
        eyeBrowPs=[]
        for i in mp_face_mesh.FACEMESH_LEFT_BROW:
            p=landmarks.landmark[i]
            eyeBrowPs.append(p)
        eyeBrowCentroid=eyeBrowPs.mean(axis=0)
        eyeBrowCentroids.append(eyeBrowCentroid)
        

        
        ###detect stress from eye blink##
        #to see if eye is blink or not we samply see if we are detecting iris or not
        if not mp_face_mesh.FACEMESH_LEFT_IRIS:
            blinks.append[1]
        else:
            blinks.append[0]
        #high frequence of blinks mean detecting stress   
        if blinks[-3:]==[0,1,0]or blinks[-3:]==[1,0,1] :
          blinkStress.append(1)
        else:
          blinkStress.append(0);        
        ###detect stress from eyes direction####
        irisPs=[]
        for j in mp_face_mesh.FACEMESH_LEFT_IRIS:
          p=landmarks.landmark[j]
          irisPs.append(p)
        irisCentroid=irisPs.mean(axis=1) 
        eyePs=[] 
        for k in mp_face_mesh.FACEMESH_LEFT_EYE:
          p=landmarks.landmark[k]
          eyePs.append(p)
        eyeCentroid=eyePs.mean(axis=1)
        eyewidh=eyePs.max(axis=1)-eyePs.min(axis=1)
        #detect wether the eye looking forward or not
        if (eyeCentroid-irisCentroid)/eyewidh>0.1:
          eyeDirectionStress.append(1)
        else:
          eyeDirectionStress.append(0)

np.var(eyeBrowCentroids,out=eyeBrowStress)
#normalize btween 0 and 1
eyeBrowStress=(eyeBrowStress-np.min(eyeBrowStress))/(np.max(eyeBrowStress)-np.min(eyeBrowStress))
#i change 0.25 to 0.33 because i work on three stress buttoms instead of 4
Stress=0.33*eyeDirectionStress+0.33*blinkStress+0.33*eyeBrowStress
x=np.arange(1,len(eyeBrowStress)+1)
plt.plot(Stress, 'r-', label='Stress')
plt.plot(eyeDirectionStress, 'g-', label='eyeDirectionStress')
plt.plot(eyeBrowStress, 'b-', label='eyeBrowStress')
plt.plot(blinkStress, 'y-', label='blinkStress')
plt.xlabel('frame')
plt.ylabel('stress')
plt.legend()
plt.show()




 
   
