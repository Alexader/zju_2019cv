import cv2
import numpy as np
from datetime import datetime

window_name = "Frame"
paintBar = None

def openCamera():
    cap = cv2.VideoCapture(0)
    if (cap.isOpened() == False):
        print("can't open camera")
    return cap

def display(cap):
    isPlaying = True
    config()

    # saveing video
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter("record.mp4", fourcc, 10.0, (1280, 720))
    while(cap.isOpened()):
        ret, frame = cap.read()
        clone = frame.copy()
        if (ret):
            # add time
            now = datetime.now()
            date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
            addTextToFrame(frame, date_time, 630)
            # add name
            name = "taoyongxing,21921105"
            addTextToFrame(frame, name, 660)

            # add logo
            logo = getLogo()
            watermark(frame, logo)

            cv2.setMouseCallback(window_name, line_drawing)
            # drawLines(frame)
            addPaintToFrame(frame)

            button = addSatrtButton()
            if (isPlaying == True):
                addTextToButton(button, "started", (255, 0, 0))
                frame[720-50:720-10, 10:110] = button
                cv2.imshow(window_name, frame)
                out.write(clone)        
                
            # Press Q on keyboard to  exit, press space to stop/play video
            key = cv2.waitKey(5) & 0xFF
            if (key == ord('q')):
                 break
            if (key == ord(' ')):
                isPlaying = not isPlaying
                if isPlaying == False:
                    button = addSatrtButton()
                    addTextToButton(button, "stopped", (0, 0, 255))
                    frame[720-50:720-10, 10:110] = button
                    cv2.imshow(window_name, frame)
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def config():
    global paintBar
    img_height = 720
    img_width = 1280
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, img_width, img_height)
    cv2.moveWindow(window_name, 200, 200)
    paintBar = np.full((img_height, img_width, 3), 180, dtype=np.uint8)

def addTextToFrame(frame, text, ycord):
    font = cv2.FONT_HERSHEY_TRIPLEX
    bottomLeftCornerOfText = (400,ycord)
    fontScale              = 1
    fontColor              = (0,0,256)
    lineType               = 2
    cv2.putText(frame, text,
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType)

def addTextToButton(button, text, color):
    font = cv2.FONT_HERSHEY_TRIPLEX
    bottomLeftCornerOfText = (10,30)
    fontScale              = 0.7
    fontColor              = color
    lineType               = 1
    cv2.putText(button, text,
            bottomLeftCornerOfText,
            font,
            fontScale,
            fontColor,
            lineType)

def getLogo():
    logo = cv2.imread("./selfie.jpg")
    scale_percent = 0.1
    resized =  cv2.resize(logo, None, fx=scale_percent, fy=scale_percent)
    return cv2.rotate(resized, cv2.ROTATE_90_CLOCKWISE)

def watermark(image, logo):
    (wH, wW) = logo.shape[:2]
    (h, w) = image.shape[:2]
    image[0:wH, w-wW:w] = logo

drawing = False # true if mouse is pressed
pt1_x , pt1_y = None , None

# mouse callback function
def line_drawing(event,x,y,flags,param):
    global pt1_x,pt1_y,drawing, paintBar

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        pt1_x,pt1_y=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            cv2.line(paintBar,(pt1_x,pt1_y),(x,y),color=(0,0,255),thickness=3)
            pt1_x,pt1_y=x,y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        cv2.line(paintBar,(pt1_x,pt1_y),(x,y),color=(0,0,255),thickness=3)

def addPaintToFrame(frame):
    # cv2.imshow("paint", paintBar)
    cv2.addWeighted(frame, 0.7, paintBar, 0.3, 0, frame)
    return

def addSatrtButton():
    shape = (40, 100, 3)
    control_image = np.full(shape, 125, np.uint8)
    # frame[h-50:h-10, 10:shape[1]+10] = control_image
    return control_image

if __name__ == "__main__":
    cap = openCamera()
    display(cap)

