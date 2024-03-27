# using MTCNN 
# Problems: 1. Very slow 
# (Different alignment detection solved)
pip install mtcnn
from mtcnn import MTCNN
import cv2

# Initialize the MTCNN detector
mtcnn_detector = MTCNN()

# Getting video live stream
video_capture = cv2.VideoCapture(0)

# Function for defining face around box
def detect_bounding_box(vid):
    # Detect faces
    detections = mtcnn_detector.detect_faces(vid)
    
    for detection in detections:
        x, y, w, h = detection['box']
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    return detections

# Loop for real-time detection
while True:
    result, video_frame = video_capture.read()  # Read frames from the video
    if not result:
        break  # Terminate the loop if the frame is not read successfully

    faces = detect_bounding_box(video_frame)  # Apply the function to the video frame

    cv2.imshow("Face Detection", video_frame)  # Display the processed frame in a window
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
