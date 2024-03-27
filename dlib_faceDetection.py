import cv2
import dlib

# Initialize the face detector from dlib
face_detector = dlib.get_frontal_face_detector()

# Getting video live stream
video_capture = cv2.VideoCapture(0)

# Function for defining face around box
def detect_bounding_box(vid):
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_detector(gray_image)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
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

