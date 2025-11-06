import cv2
import mediapipe as mp

class ExpressionRecognizer:
    def __init__(self, smileThreshold = 0.35):
        self.mpFaceMesh = mp.solutions.face_mesh
        self.mpHands = mp.solutions.hands
        self.smileThreshold = smileThreshold
        self.headZone = {"yMin": 0.0, "yMax": 0.35}
        self.mouthZoneSize = 0.15

        self.faceMesh = self.mpFaceMesh.FaceMesh(
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5
        )
        self.hands = self.mpHands.Hands(
            min_detection_confidence = 0.5,
            min_tracking_confidence = 0.5
        )


    def getHandPosition(self, imageRGB):
        """
        Gets the positions of the hands by calculating the index, and middle finger tips, and the positions of the wrist
        Returns: the positions of the wrists
        """
        resultHands = self.hands.process(imageRGB)
        handPositions = []

        if resultHands.multi_hand_landmarks:
            for handLandmarks in resultHands.multi_hand_landmarks:
                indexTip = handLandmarks.landmark[self.mpHands.HandLandmark.INDEX_FINGER_TIP]
                middleTip = handLandmarks.landmark[self.mpHands.HandLandmark.MIDDLE_FINGER_TIP]
                wrist = handLandmarks.landmark[self.mpHands.HandLandmark.WRIST]

                handPositions.append({
                    "indexTip": indexTip,
                    "middleTip": middleTip,
                    "wrist": wrist
                })
        return handPositions
    
    def getMouthZone(self, imageRGB):
        """
        Calculates the area of the mouth/thinking zone
        Returns: the mouth zone
        """
        resultFace = self.faceMesh.process(imageRGB)

        if not resultFace.multi_face_landmarks:
            return None
        
        faceLandmarks = resultFace.multi_face_landmarks[0]

        # Mouth landmarks
        leftCorner = faceLandmarks.landmark[291]
        rightCorner = faceLandmarks.landmark[61]
        upperLip = faceLandmarks.landmark[13]
        bottomLip = faceLandmarks.landmark[14]

        # Create zone around the mouth
        mouthCenterX = (leftCorner.x + rightCorner.x) / 2
        mouthCenterY = (upperLip.y + bottomLip.y) / 2

        mouthZone = {
            "centerX": mouthCenterX,
            "centerY": mouthCenterY,
            "xMin": mouthCenterX - self.mouthZoneSize,
            "xMax": mouthCenterX + self.mouthZoneSize,
            "yMin": mouthCenterY - self.mouthZoneSize,
            "yMax": mouthCenterY + self.mouthZoneSize
        }

        return mouthZone
    
    def checkHandInZone(self, handPos, zone):
        """
        Checks if the positions of of the index or middle finger tips are in the zones
        Returns: indexIn or middleIn
        """
        if zone is None:
            return False
        
        indexIn = (zone["xMin"] <= handPos["indexTip"].x <= zone["xMax"] and 
                   zone["yMin"] <= handPos["indexTip"].y <= zone["yMax"])
        middleIn = (zone["xMin"] <= handPos["middleTip"].x <= zone["xMax"] and
                     zone["yMin"] <= handPos["middleTip"].y <= zone["yMax"])

        return indexIn or middleIn
    
    def checkHandsUp(self, handPositions):
        """
        Check if the hands are in the hands up zone
        Returns: True if they are and false if they are not
        """
        for handPos in handPositions:
            if (handPos["wrist"].y < self.headZone["yMax"] or
                handPos["indexTip"].y < self.headZone["yMax"] or 
                handPos["middleTip"].y < self.headZone["yMax"]):
                return True, handPositions
        return False, handPositions
    
    def checkThinkingPose(self, handPositions, mouthZone):
        """
        Checks if the user is thinking, does this by checking in index or middle finger tips are in the mouthzone
        Returns: True if they are in and false if they are not
        """
        if mouthZone is None:
            return False
        
        for handPos in handPositions:
            if self.checkHandInZone(handPos, mouthZone):
                return True
        return False
    
    def analyzeFace(self, imageRGB):
        """
        Analyzes the face through measuring the mouth and eye landmarks. Does this to try to recognize emotions
        Returns: The matching emotion to the face the user is making, if it can't recognize it it will return a blank emotion
        """
        resultFace = self.faceMesh.process(imageRGB)

        if not resultFace.multi_face_landmarks:
            return "STRAIGHTFACE", None, None
        
        faceLandmarks = resultFace.multi_face_landmarks[0]

        # Mouth landmarks
        leftCorner = faceLandmarks.landmark[291]
        rightCorner = faceLandmarks.landmark[61]
        upperLip = faceLandmarks.landmark[13]
        bottomLip = faceLandmarks.landmark[14]

        # Eye landmarks
        leftEyeTop = faceLandmarks.landmark[159]
        leftEyeBottom = faceLandmarks.landmark[145]
        rightEyeTop = faceLandmarks.landmark[386]
        rightEyeBottom = faceLandmarks.landmark[374]

        # Calculate dimensions
        mouthWidth = ((rightCorner.x - leftCorner.x) ** 2 + (rightCorner.y - leftCorner.y) ** 2) ** 0.5
        mouthHeight = abs(bottomLip.y - upperLip.y)
        leftEyeOpen = abs(leftEyeTop.y - leftEyeBottom.y)
        rightEyeOpen = abs(rightEyeTop.y - rightEyeBottom.y)
        avgEyeOpen = (leftEyeOpen + rightEyeOpen) / 2
        mouthCenterX = (leftCorner.x + rightCorner.x) / 2
        mouthCenterY = (upperLip.y + bottomLip.y) / 2

        mouthZone = {
            "centerX": mouthCenterX,
            "centerY": mouthCenterY,
            "xMin": mouthCenterX - self.mouthZoneSize,
            "xMax": mouthCenterX + self.mouthZoneSize,
            "yMin": mouthCenterY - self.mouthZoneSize,
            "yMax": mouthCenterY + self.mouthZoneSize
        }

        if mouthWidth > 0:
            mouthAspectRatio = mouthHeight / mouthWidth

            if mouthAspectRatio > self.smileThreshold and mouthWidth > 0.05:
                return "SMILING", mouthZone, None
            
            elif bottomLip.y > upperLip.y + 0.01 and avgEyeOpen < 0.02:
                return "SAD", mouthZone, None
            
            elif mouthWidth < 0.12 and mouthAspectRatio < 0.2:
                return "ANGRY", mouthZone, None
        return "STRAIGHTFACE", mouthZone, None
    
    def getState(self, imageRGB):
        """
        Checks if the user is either doing hands up or thinking
        Returns: hands up or thinking state
        """
        handPositions = self.getHandPosition(imageRGB)
        handsUp, handData = self.checkHandsUp(handPositions)
        mouthZone = self.getMouthZone(imageRGB)

        vizualData = {
            "handsUp": handsUp,
            "handPositions": handPositions,
            "mouthZone": mouthZone,
            "headZone": self.headZone
        }

        if handsUp:
            return "HANDSUP", vizualData
        
        thinking = self.checkThinkingPose(handPositions, mouthZone)
        if thinking:
            return "CONFUSED", vizualData
        
        state, mouthZone, _ = self.analyzeFace(imageRGB)
        vizualData["mouthZone"] = mouthZone
        return state, vizualData
    
    def release(self):
        self.faceMesh.close()
        self.hands.close()