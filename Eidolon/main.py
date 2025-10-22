import cv2, os
import mediapipe as mp
from settings import Config

config = Config()

# Initializing Mediapipe
mpPose = mp.solutions.pose
mpFaceMesh = mp.solutions.face_mesh
mpDrawing = mp.solutions.drawing_utils

# Setting from Config
smileTreshold = config.SMILETRESHOLD
windowWidth = config.WINDOWWIDTH
windowHeight = config.WINDOWHEIGHT
PibbleWindowSize = config.PIBBLEWINDOWSIZE

# Initialize the directories
currentDIR = os.path.dirname(os.path.abspath(__file__))
pibbleDIR = os.path.join(currentDIR, "Pibbles")

# Try to load the Pibble images
try:
    smilingPibble = cv2.imread(os.path.join(pibbleDIR, "happyPibble.jpeg"))
    confusedPibble = cv2.imread(os.path.join(pibbleDIR, "confusedPibble.jpeg"))
    sadPibble = cv2.imread(os.path.join(pibbleDIR, "sadPibble.jpeg"))
    angryPibble = cv2.imread(os.path.join(pibbleDIR, "angryPibble.jpeg"))
    handsUpPibble = cv2.imread(os.path.join(pibbleDIR, "HandsUpPibble.jpeg"))
    blankPibble = cv2.imread(os.path.join(pibbleDIR, "Pibble.jpeg"))

    if smilingPibble is None:
        raise FileNotFoundError("Smiling Pibble image not found.")
    if confusedPibble is None:
        raise FileNotFoundError("Confused Pibble image not found.")
    if sadPibble is None:
        raise FileNotFoundError("Sad Pibble image not found.")
    if angryPibble is None:
        raise FileNotFoundError("Angry Pibble image not found.")
    if handsUpPibble is None:
        raise FileNotFoundError("Hands up Pibble image not found.")
    if blankPibble is None:
        raise FileNotFoundError("Blank Pibble image not found.")
    
    smilingPibble = cv2.resize(smilingPibble, PibbleWindowSize)
    confusedPibble = cv2.resize(confusedPibble, PibbleWindowSize)
    sadPibble = cv2.resize(sadPibble, PibbleWindowSize)
    angryPibble = cv2.resize(angryPibble, PibbleWindowSize)
    handsUpPibble = cv2.resize(handsUpPibble, PibbleWindowSize)
    blankPibble = cv2.resize(blankPibble, PibbleWindowSize)


except Exception as e:
    print("Error loading Pibble images")
    print(f"Details: {e}")
    exit()

# Initializing webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

cv2.namedWindow("Eidolon Feed", cv2.WINDOW_NORMAL)
cv2.namedWindow("Pibble Output", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Eidolon Feed", windowWidth, windowHeight)
cv2.resizeWindow("Pibble Output", windowWidth, windowHeight)
cv2.moveWindow("Pibble Output", windowWidth + 150, 100)

with mpPose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose, \
    mpFaceMesh.FaceMesh(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as faceMesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imageRGB.flags.writeable = False
        currentState = "STRAIGHTFACE"

        # Checks for hands up
        resultPose = pose.process(imageRGB)
        if resultPose.pose_landmarks:
            landmarks = resultPose.pose_landmarks.landmark

            leftShoulder = landmarks[mpPose.PoseLandmark.LEFT_SHOULDER]
            rightShoulder = landmarks[mpPose.PoseLandmark.RIGHT_SHOULDER]
            leftWrist = landmarks[mpPose.PoseLandmark.LEFT_WRIST]
            rightWrist = landmarks[mpPose.PoseLandmark.RIGHT_WRIST]

            if (leftWrist.y < leftShoulder.y) or (rightWrist.y < rightShoulder.y):
                currentState = "HANDSUP"
        
        # Checks for facial expressions incase the hands are not up
        if currentState != "HANDSUP":
            resultFace = faceMesh.process(imageRGB)
            if resultFace.multi_face_landmarks:
                for faceLandmarks in resultFace.multi_face_landmarks:
                    # Sets the landmarks for the lips
                    leftCorner = faceLandmarks.landmark[291]
                    rightCorner = faceLandmarks.landmark[61]
                    upperLip = faceLandmarks.landmark[13]
                    bottomLip = faceLandmarks.landmark[14]

                    # Sets the landmarks for the eyebrows
                    leftEyebrowInner = faceLandmarks.landmark[107]
                    rightEyebrowInner = faceLandmarks.landmark[336]
                    leftEyebrowOuter = faceLandmarks.landmark[66]
                    rightEyebrowOuter = faceLandmarks.landmark[296]

                    # Sets the landmarks for the eyes
                    leftEyeTop = faceLandmarks.landmark[159]
                    leftEyeBottom = faceLandmarks.landmark[145]
                    rightEyeTop = faceLandmarks.landmark[386]
                    rightEyeBottom = faceLandmarks.landmark[374]

                    # calculates thes mouth dimensions
                    mouthWidth = ((rightCorner.x - leftCorner.x) ** 2 + (rightCorner.y - leftCorner.y) ** 2) ** 0.5
                    mouthHeight = abs(bottomLip.y - upperLip.y)

                    # Calculates the eyebrow angle
                    leftEyebrowHeight = leftEyebrowInner.y
                    rightEyebrowHeight = rightEyebrowInner.y
                    eyebrowAngle = abs(leftEyebrowHeight - rightEyebrowHeight)

                    # checks if the eyes are open and how open they are
                    leftEyeOpen = abs(leftEyeTop.y - leftEyeBottom.y)
                    rightEyeOpen = abs(rightEyeTop.y - rightEyeBottom.y)
                    avgEyeOpen = (leftEyeOpen + rightEyeOpen) / 2

                    if mouthWidth > 0:
                        mouthAspectRatio = mouthHeight / mouthWidth

                        if mouthAspectRatio > smileTreshold and mouthWidth > 0.05:
                            currentState = "SMILING"
                        elif bottomLip.y > upperLip.y + 0.01 and avgEyeOpen < 0.02:
                            currentState = "SAD"
                        elif leftEyebrowHeight < 0.3 and rightEyebrowHeight < 0.3 and mouthWidth < 0.12:
                            currentState = "ANGRY"
                        elif eyebrowAngle > 0.02 and 0.1 < mouthAspectRatio < 0.25:
                            currentState = "CONFUSED"
                        else:
                            currentState = "STRAIGHTFACE"
        
        # Select the Pibble based on the state
        if currentState == "SMILING":
            pibbleToShow = smilingPibble
            pibbleName = "Smiling Pibble"
        elif currentState == "HANDSUP":
            pibbleToShow = handsUpPibble
            pibbleName = "Hands Up Pibble"
        elif currentState == "SAD":
            pibbleToShow = sadPibble
            pibbleName = "Sad Pibble"
        elif currentState == "ANGRY":
            pibbleToShow = angryPibble
            pibbleName = "Angry Pibble"
        elif currentState == "CONFUSED":
            pibbleToShow = confusedPibble
            pibbleName = "Confused Pibble"
        else:
            pibbleToShow = blankPibble
            pibbleName = "Pibble :)"
        
        cameraFrameResized = cv2.resize(frame, (windowWidth, windowHeight))
        cv2.putText(cameraFrameResized, f"STATE: {pibbleName}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(cameraFrameResized, "Press 'esc' to exit", (10, windowHeight - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("Eidolon Feed", cameraFrameResized)
        cv2.imshow("Pibble Output", pibbleToShow)

        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()