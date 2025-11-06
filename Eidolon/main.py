import cv2, os
import mediapipe as mp
from settings import Config
from recognizer import ExpressionRecognizer
from visualizer import Visualizer

def loadPibbleImages(pibbleDIR, windowSize):
    """
    Loads the images and prints out errors incase it can't load the images
    Returns: Images
    """
    images = {}
    imageFiles = {
        "smiling": "happyPibble.jpeg",
        "confused":"confusedPibble.jpeg",
        "sad": "sadPibble.jpeg",
        "angry": "angryPibble.jpeg",
        "handsUp": "HandsUpPibble.jpeg",
        "blank": "Pibble.jpeg"
    }

    try:
        for key, filename in imageFiles.items():
            path = os.path.join(pibbleDIR, filename)
            img = cv2.imread(path)
            if img is None:
                raise FileNotFoundError(f"{key.title()} Pibble image is not found.")
            images[key] = cv2.resize(img, windowSize)
        return images
    except Exception as e:
        print("Error loading Pibble images")
        print(f"Details: {e}")
        exit()

def getPibbleForState(state, pibbleImages):
    """
    Sets the right Pibble per state of the user
    Returns: pibbleImage, pibbleName
    """
    stateMapping = {
        "SMILING": ("smiling", "Smiling Pibble"),
        "HANDSUP": ("handsUp", "Hands Up Pibble"),
        "SAD": ("sad", "Sad Pibble"),
        "ANGRY": ("angry", "Angry Pibble"),
        "CONFUSED": ("confused", "Confused Pibble"),
        "STRAIGHTFACE": ("blank", "Pibble :)")
    }
    key, name = stateMapping.get(state, ("blank", "Pibble :)"))
    return pibbleImages[key], name

def main():
    config = Config()
    currentDIR = os.path.dirname(os.path.abspath(__file__))
    pibbleDIR = os.path.join(currentDIR, "Pibbles")
    pibbleImages = loadPibbleImages(pibbleDIR, config.PIBBLEWINDOWSIZE)
    recognizer = ExpressionRecognizer(smileThreshold = config.SMILETRESHOLD)
    visualizer = Visualizer(config.WINDOWWIDTH, config.WINDOWHEIGHT)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        exit()

    cv2.namedWindow("Eidolon Feed", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Pibble Output", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Eidolon Feed", config.WINDOWWIDTH, config.WINDOWHEIGHT)
    cv2.resizeWindow("Pibble Output", config.WINDOWWIDTH, config.WINDOWHEIGHT)
    cv2.moveWindow("Pibble Output", config.WINDOWWIDTH + 150, 100)

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)
            imageRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imageRGB.flags.writeable = False

            currentState, visualData = recognizer.getState(imageRGB)
            pibbleToShow, pibbleName = getPibbleForState(currentState, pibbleImages)
            cameraFrameResized = cv2.resize(frame, (config.WINDOWWIDTH, config.WINDOWHEIGHT))
            cameraFrameResized = visualizer.drawDetectionBoxes(cameraFrameResized, currentState, visualData)
            cameraFrameResized = visualizer.addUiElements(cameraFrameResized, pibbleName)
            cv2.imshow("Eidolon Feed", cameraFrameResized)
            cv2.imshow("Pibble Output", pibbleToShow)

            if cv2.waitKey(5) & 0xFF == 27:
                break
    finally:
        recognizer.release()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()