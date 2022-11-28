#Import all modules:
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands #For detecting the hands
hands = mpHands.Hands() #Hands parameters: static_image_mode= False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5


mpDraw = mp.solutions.drawing_utils #For drawing the landmarks.

#FPS initialization:
pTime = 0 #Previous time
cTime = 0 #Current time
###

while True:
    success, img = cap.read()

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #The hands object only uses RGB images.
    results = hands.process(imgRGB) #The process method of hands object processes all the image frames and provides the final result.
    # print(results.multi_hand_landmarks) #To print whether hand landmarks getting detected or not.
    
    if results.multi_hand_landmarks:                    #If the results is true, execute this.
        for handLms in results.multi_hand_landmarks:    #For each hand Landmarks, execute this.
            for id, lm in enumerate(handLms.landmark):  #ID represents the specific points detected on each hand and lm represents the x,y,z coordinates of those hand ID points.
                print(id,lm)
                h, w, c = img.shape #Getting height, width and channel of image.
                cx, cy = int(lm.x*w), int(lm.y*h) #cx and cy show the center x and y coordinates
                if id == 4: #Showing one index differently than the rest.
                    cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)         #Drawing and connecting all the landmark points of each hand.
    
    #FPS calculation and display works:
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    
    cv2.putText(img, str(int((fps))), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
    ###

    cv2.imshow('Image', img)
    cv2.waitKey(1)