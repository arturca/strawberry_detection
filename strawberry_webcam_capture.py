import cv2
import numpy as np

# CONST
chanel1_min_green = int(0.08 * 255)
chanel1_max_green = int(0.2 * 255)

chanel2_min_green = int(0.15 * 255)
chanel2_max_green = int(0.65 * 255)

channel3_min_green = int(0.35 * 255)
channel3_max_green = int(1.0 * 255)

lower_green = np.array([chanel1_min_green, chanel2_min_green, channel3_min_green])
upper_green = np.array([chanel1_max_green, chanel2_max_green, channel3_max_green])

channel1_min_red = int(0.5 * 255)
channel1_max_red = int(1.0 * 255)

channel2_min_red = int(0.20 * 255)
channel2_max_red = int(0.8 * 255)

channel3_min_red = int(0.45 * 255)
channel3_max_red = int(1.0 * 255)

lower_red = np.array([channel1_min_red, channel2_min_red, channel3_min_red])
upper_red = np.array([channel1_max_red, channel2_max_red, channel3_max_red])


# END CONST


def create_mask(frame):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask_green = cv2.inRange(hsv_frame, lower_green, upper_green)
    mask_red = cv2.inRange(hsv_frame, lower_red, upper_red)

    # GREEN
    mask_green = cv2.erode(mask_green, None, iterations=3)
    mask_green = cv2.dilate(mask_green, None, iterations=4)

    # RED
    mask_red = cv2.erode(mask_red, None, iterations=3)
    mask_red = cv2.dilate(mask_red, None, iterations=4)
    mask =  mask_green | mask_red

    return mask



# video = cv2.VideoCapture(0)
video = cv2.VideoCapture('ip_webcam_link_end_at__video')

i=0
while True:
    i += 1
    check, frame = video.read()
    # mask =0
    mask=create_mask(frame)

    contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        for c in contours:
            if 400 < cv2.contourArea(c) < 3000:
                (x,y,w,h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 3)

    cv2.imshow('Strawberry Vision', frame)
    kay=cv2.waitKey(50)
    if i == 20000:
        break
video.release()
cv2.destroyAllWindows()