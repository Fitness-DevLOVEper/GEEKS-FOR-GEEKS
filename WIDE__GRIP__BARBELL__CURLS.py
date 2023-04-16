import mediapipe as mp
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def calculate_angle(a,b,c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    angR = np.arctan2(c[1]-b[1],c[0]-b[0]) - np.arctan2(a[1]-b[1],a[0]-b[0])
    angD = np.abs(angR*180.0/np.pi)

    if angD>180:
        angD = 360 - angD
        return angD
    else:
        return angD

def counter(angle):
    global count
    global stage
    if angle > 160:
        stage = "down"
    if angle < 30 and stage == "down":
        stage = "up"
        count = count + 1
        print("Count :", count)
    return count

def face_detect():
    detected = 0
    col=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces= face_cap.detectMultiScale(
        col, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for(x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
        detected = 1
    if detected == 1:
        return 1
    else:
        return 0
cap = cv2.VideoCapture(0)
face_cap = cv2.CascadeClassifier(r"C:\Users\Kongara Bhargavaram\AppData\Roaming\Python\Python39\site-packages\cv2\data\haarcascade_frontalface_default.xml")
stage = None
count=0
while True:

    suc,img = cap.read()
    k = face_detect()
    if k==1:
        with mp_pose.Pose(min_detection_confidence=0.6,min_tracking_confidence=0.6,static_image_mode=True) as pose:
            ret, frame = cap.read()
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img.flags.writeable = False

            results = pose.process(img)
            # re-colouring back to BGR from RGB
            img.flags.writeable = True
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            try:
                print("In try block")
                landmarks = results.pose_landmarks.landmark
            # GET COORDINATES
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

                rangle = calculate_angle(rshoulder, relbow, rwrist)
                langle = calculate_angle(lshoulder, lelbow, lwrist)

                rw = np.array(rwrist)  # rw(x1,y1) = ( rw[0],rw[1] )
                lw = np.array(lwrist)  # lw(x2,y2) = ( lw[0],lw[1] )

                ''' FOR  CALCULATING  WIDE  GRIP  CURLS  THE  MINIMUM   DISTANCE  CAN  VARY  FOR  INDIVIDUAL 
                    BASED  ON  HIS/HER  ANATOMY. BUT  GENERAL  IS CONSIDERED  AS VARIES  FROM  40-50cms.
                    
                    dist  =  sqroot((x2-x1)**2 + (y2-y1)**2)'''
                rdist = (lw[0]-rw[0])**2
                ldist = (lw[1]-rw[1])**2
                dist = np.sqrt((rdist+ldist))
                print("Distance :",dist*100)
                cv2.putText(img, str(int(rangle)),tuple(np.multiply(relbow, [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(img, str(int(langle)),tuple(np.multiply(lelbow, [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2, cv2.LINE_AA)

                both_hands = 0
                dist_counter = 0
             # counter logic
                if ((langle-50)<=rangle<=(langle+50)):
                    both_hands = 1     # ang_error = 1 implies no error
                    print("dist_counter =",dist_counter)
                    print("both_hands =",both_hands)
                    if(dist*100>=40):
                        dist_counter = 1
                        print("IN if OF DIST")
                        x = counter(int(rangle))
                        print("Counter =",x)
                        print("===============================")
                        cv2.rectangle(img, (420, 0), (640, 73), (0, 0, 0), -1)
                        cv2.putText(img, 'BARBELL CURLS', (420, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        cv2.putText(img, 'REP(S) :', (420, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1)
                        cv2.putText(img, str(x), (560, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                else:
                    # Now to check exactly which one we got as 0 so as to print it
                    if(both_hands==0):
                        cv2.rectangle(img, (60, 20), (600, 220), (0, 0, 0), -1)
                        cv2.putText(img,"Please use both ",(70,100),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255),2)
                        cv2.putText(img,"the hands.",(140,180),cv2.FONT_HERSHEY_SIMPLEX,2,(0,255,255),2)
                        print("Please Use both the hands")

            except():
                print("In except :")
                pass
            except(AttributeError):
                print("Attribute error")
                pass
            except(KeyboardInterrupt):
                print("Keyboard Interrupt error")
                pass

            mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245,117,66),thickness=2,circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245,66,230),thickness=2,circle_radius=2))

            cv2.imshow("Barbell-Bicep", img)
            if cv2.waitKey(1) & 0XFF == ord('q'):
                break
    else:
        cv2.rectangle(img, (30, 120), (640, 200), (0, 0, 0), -1)
        cv2.putText(img, " PLEASE BE VISIBLE ", (1, 180), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 255, 255), 2)
        # TO CAMERA FOR COUNTING YOUR REPS
        cv2.rectangle(img, (70, 201), (610, 280), (0, 0, 0), -1)
        cv2.putText(img, " TO THE CAM FOR  ", (40, 260), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 255, 255), 2)
        cv2.rectangle(img, (60, 280), (600, 350), (0, 0, 0), -1)
        cv2.putText(img, " COUNTING YOUR ", (40, 330), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 255, 255), 2)
        cv2.rectangle(img, (230, 342), (420, 420), (0, 0, 0), -1)
        cv2.putText(img, " REPS. ", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 2,
                    (0, 255, 255), 2)
        cv2.imshow("Barbell-Bicep",img)
        if cv2.waitKey(1) & 0XFF==ord('q'):
            break