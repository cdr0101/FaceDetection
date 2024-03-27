# using haar cascade classifier 
# Problems: 1. False positives  
#           2. Doesn't detect for different face alignments
import cv2

# Load the pre-trained face detector
face_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Getting video live stream
video_capture = cv2.VideoCapture(0)

# Function for defining face around box
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_detector.detectMultiScale(gray_image, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return faces

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


# cmd to start: python .\filename.py
# press q to quit