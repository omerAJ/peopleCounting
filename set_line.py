## use this code to set coordinates of the line. It will open stream, 
# you click on where you want to get coords of. 
# The coords are printed on screen just use those.

import cv2

# Replace the URL with your RTSP stream URL
RTSP = ""
rtsp_url = f"rtsp://user:pass@{RTSP}"
vid = 'D:\\omer\\Human_counting_project\\gateVideos\\gate2.mp4'
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print("Coordinates: X: ", x, " Y: ", y)

cap = cv2.VideoCapture(vid)

cv2.namedWindow("RTSP Stream", cv2.WINDOW_NORMAL)
cv2.resizeWindow("RTSP Stream", (1920, 1080))
cv2.setMouseCallback("RTSP Stream", click_event)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    cv2.imshow("RTSP Stream", frame)

    # Break the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



# gate1 == Coordinates: X:  1252  Y:  551
#          Coordinates: X:  2404  Y:  585

# gate2 == Coordinates: X:  579  Y:  339
#          Coordinates: X:  2300  Y:  1220