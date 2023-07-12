import cv2
import mediapipe as mp

# py -3.10 -m venv venv
# pip install opencv-python
# pip install mediapipe

# Create a MediaPipe FaceMesh object
face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Open the webcam
video = cv2.VideoCapture(0)

while True:
    # Read the webcam frame
    ret, frame = video.read()

    if not ret:
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe FaceMesh
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for landmark in face_landmarks.landmark:
                # Get the x, y coordinates of the landmark
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                # Draw a circle on the face landmark
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Show the frame with landmarks
    cv2.imshow("Face Landmarks", frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the windows
video.release()
cv2.destroyAllWindows()

