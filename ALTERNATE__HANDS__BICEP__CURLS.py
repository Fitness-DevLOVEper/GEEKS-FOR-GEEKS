import mediapipe as mp
import cv2
import numpy as np
 # start curls with your left hand
def calc_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    rad = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angD = np.abs(rad*180.0/np.pi)

    if angD>180:
        angD = 360 - angD
        return angD
    else:
        return angD

def detect():
    detected = 0
    col = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(
        col, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        detected = 1 
    if detected==1:
        return detected
    else:
        # cv2.putText(img, " PLEASE BE VISIBLE TO CAMERA TO COUNT YOUR REPS", (1, 74), cv2.FONT_HERSHEY_SIMPLEX, 2,(0, 255, 255), 2)
        return 0

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
face_cap = cv2.CascadeClassifier(r"C:\Users\Kongara Bhargavaram\AppData\Roaming\Python\Python39\site-packages\cv2\data\haarcascade_frontalface_default.xml")
cap = cv2.VideoCapture(0)# number represents web-cam
lcounter = 0 # to count left arm reps
rcounter = 0 # to count right arm reps
c = 0

while True:
    succ,img = cap.read()
    k=detect()
    if k==1:
        with mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7,static_image_mode=True) as pose:
            ret, frame = cap.read()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False

            # actually making detections
            results = pose.process(img)

            # re-colouring back to BGR from RGB
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            try:
                landmarks = results.pose_landmarks.landmark
                # GET CO-ORDINATES
                rshoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x,
                             landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y]
                relbow = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y]
                rwrist = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                          landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y]

                lshoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y]
                lelbow = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y]
                lwrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                      landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]

                # CALCULATE ANGLE
                rangle = calc_angle(rshoulder, relbow, rwrist)
                langle = calc_angle(lshoulder, lelbow, lwrist)

                # BUILDING COUNTER LOGIC LEFT ARM
                if (c == 0):
                    if langle > 160:
                        stage = "down"
                    if langle < 20 and stage == "down":
                        stage = "up"
                        lcounter = lcounter + 1
                        c = 1
                        print("Left Counter :", lcounter)
                         # gggggggg

                # BUILDING COUNTER LOGIC RIGHT ARM
                if (c == 1):
                    if rangle > 160:
                        stage = "down"
                    if rangle < 20 and stage == "down":
                        stage = "up"
                        rcounter = rcounter + 1
                        c = 0
                        print("Right Counter :", rcounter)

                # PUTTING COUNT ON SCREEN FOR LEFT ARM
                cv2.rectangle(img, (420, 0), (640, 73), (0, 0, 0), -1)
                cv2.putText(img, 'LEFT ARM', (420, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(img, 'REP(S) :', (420, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                cv2.putText(img, str(lcounter), (560, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)

                # PUTTING COUNT ON SCREEN FOR RIGHT ARM

                cv2.rectangle(img, (0, 0), (225, 73), (0, 0, 0), -1)
                cv2.putText(img, 'RIGHT ARM', (0, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(img, 'REP(S) :', (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                cv2.putText(img, str(rcounter), (140, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
            except:
                print("In except")
                pass

            # Rendering detection or actually detecting
            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                        mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
            ''' First one in Drawing_spec is for joints
                Second one is for connections(bones)'''
            cv2.imshow(" BICEP CURLS ",img)
            if cv2.waitKey(1) == ord('q'):
                break
    else:
        cv2.rectangle(img, (30, 120), (640, 200), (0, 0, 0), -1)
        cv2.putText(img, " PLEASE BE VISIBLE ", (1, 180), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 255, 255), 2)
        #TO CAMERA FOR COUNTING YOUR REPS
        cv2.rectangle(img, (70,201), (610, 280), (0, 0, 0), -1)
        cv2.putText(img, " TO THE CAM FOR  ", (40, 260), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 255, 255), 2)
        cv2.rectangle(img, (60, 280), (600, 350), (0, 0, 0), -1)
        cv2.putText(img, " COUNTING YOUR ", (40, 330), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 255, 255), 2)
        cv2.rectangle(img, (230, 342), (420, 420), (0, 0, 0), -1)
        cv2.putText(img, " REPS. ", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 255, 255), 2)
        cv2.imshow(" BICEP CURLS ", img)
        if cv2.waitKey(30) == ord('q'):
            break
cap.release()
print("========================")
print("Total left arm count :",lcounter)
print("Total right arm count :",rcounter)
print("Total Count :",(lcounter+rcounter))
print("========================")