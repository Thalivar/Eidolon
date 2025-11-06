import cv2

class Visualizer:
    def __init__(self, windowWidth, windowHeight):
        self.windowWidth = windowWidth
        self.windowHeight = windowHeight

    def drawDetectionBoxes(self, frame, state, vizualData):
        """
        Draws both the detection boxes for the hands up zone and thinking zone
        Returns: the frame with the boxes
        """
        height, width = frame.shape[:2]

        if state == "CONFUSED" and vizualData["mouthZone"]:
            mouthZone = vizualData["mouthZone"]
            xMin = int(mouthZone["xMin"] * width)
            xMax = int(mouthZone["xMax"] * width)
            yMin = int(mouthZone["yMin"] * height)
            yMax = int(mouthZone["yMax"] * height)

            cv2.rectangle(frame, (xMin, yMin), (xMax, yMax), (0, 255, 255), 2)
            cv2.putText(frame, "Thinking Zone", (xMin, yMin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if vizualData["handPositions"]:
                for handPos in vizualData["handPositions"]:
                    indexX = int(handPos["indexTip"].x * width)
                    indexY = int(handPos["indexTip"].y * height)
                    cv2.circle(frame, (indexX, indexY), 8, (255, 0, 255), -1)
                    cv2.circle(frame, (indexX, indexY), 10, (0, 255, 255), 2)
        
        elif state == "HANDSUP":
            headZone = vizualData["headZone"]
            zoneYMax = int(headZone["yMax"] * height)
            cv2.rectangle(frame, (0, 0), (width, zoneYMax), (0, 255, 0), 3)
            cv2.putText(frame, "Hands Up Zone", (width // 2 - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            if vizualData["handPositions"]:
                for handPos in vizualData["handPositions"]:
                    indexX = int(handPos["indexTip"].x * width)
                    indexY = int(handPos["indexTip"].y * height)
                    wristX = int(handPos["wrist"].x * width)
                    wristY = int(handPos["wrist"].y * height)

                    cv2.line(frame, (wristX, wristY), (indexX, indexY), (128, 0, 128), 2)
        
        else:
            headZone = vizualData["headZone"]
            zoneYMax = int(headZone["yMax"] * height)
            cv2.line(frame, (0, zoneYMax), (width, zoneYMax), (100, 100, 100), 1)
            cv2.putText(frame, "Head Zone", (10, zoneYMax - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)

            if vizualData["mouthZone"]:
                mouthZone = vizualData["mouthZone"]
                xMin = int(mouthZone["xMin"] * width)
                xMax = int(mouthZone["yMax"] * width)
                yMin = int(mouthZone["yMin"] * height)
                yMax = int(mouthZone["yMax"] * height)

                cv2.rectangle(frame, (xMin, yMin), (xMax, yMax), (100, 100 ,100), 1)
                cv2.putText(frame, "Mouth Zone", (xMin, yMax + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 100), 1)
        
        return frame
    
    def addUiElements(self, frame, pibbleName):
        cv2.putText(frame, f"STATE: {pibbleName}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Press 'esc' to exit", (10, self.windowHeight - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        return frame