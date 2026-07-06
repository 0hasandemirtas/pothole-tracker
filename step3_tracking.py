from ultralytics import YOLO
import cv2


model_path="best.pt"
video_path="video.mp4"

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()

    if success:

        results = model.track(frame, persist=True, tracker="bytetrack.yaml")

        annotated_frame = results[0].plot()

        cv2.imshow("YOLO26 Tracking", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
