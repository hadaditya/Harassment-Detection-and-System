# import necessary packages

import cv2
import numpy as np
from playsound import playsound
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
import smtplib
import imghdr

# mail the police station
def mailfunc(i):
    from email.message import EmailMessage
    Sender_Email = "adiabhi2024@gmail.com"                    
    Reciever_Email = "hadaditya@gmail.com"
    Password = "nopy@!@#"
    newMessage = EmailMessage()                         
    newMessage['Subject'] = "Harassment Detected" 
    newMessage['From'] = Sender_Email                   
    newMessage['To'] = Reciever_Email                   
    newMessage.set_content('Here are the pictures from the scene!') 

    files = []
    for x in range(0, i):
        files.append(f"{x}.jpg")

    print(files)

    for file in files:
        with open(file, 'rb') as f:
            file_data = f.read()
            file_name = f.name
        newMessage.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        
        smtp.login(Sender_Email, Password)              
        smtp.send_message(newMessage)

# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

#trained data of Face Detector
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)


# Initialize the webcam
cap = cv2.VideoCapture(0)     

kill = 0
count = 0
while True:
    # Read each frame from the webcam
    isTrue, frame = cap.read()   

    x, y, c = frame.shape

    # Flip the frame vertically
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #convert to grayscale
    grayscaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    # Get hand landmark prediction
    result = hands.process(framergb)

    #get face coordinates
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    # print(result)
    
    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                # print(id, lm)
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)

                landmarks.append([lmx, lmy])

            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks])
            # print(prediction)
            classID = np.argmax(prediction)
            className = classNames[classID]
    
    #iterate over the faces
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame,(x,y),(x+w,y+h), (0,255,0),2)

    # show the prediction on the frame
    cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0,0,255), 2, cv2.LINE_AA) 

    
    if(className == 'victim'):
        if count <= 10:
            for (x,y,w,h) in face_coordinates:
                face = frame[y:y+h, x:x+w] #slice the face from the image
                cv2.imwrite(str(count)+'.jpg', face) #save the image
                count = count + 1
                cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        
    if(className == 'Contacting Nearby Police Station'):
        if(kill == 0):
            mailfunc(count)
            kill = kill + 1
        
            
    #if(sound):
    #   playsound('audio.mp3')

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()

cv2.destroyAllWindows()