import cv2
from ultralytics import YOLO

def main():
    # Load the dedicated face detection model
    model_path = 'models/yolov8n-face.pt'
    print(f"Loading model: {model_path}...")
    model = YOLO(model_path)

    # Initialize webcam (0 is default webcam)
    # Alternatively, provide a video file path, e.g., 'path/to/video.mp4'
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video stream or camera.")
        return

    print("Starting video stream... Press 'q' to exit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break

        # Perform inference on the frame
        # conf=0.5 sets the minimum confidence threshold for detections
        results = model.predict(source=frame, conf=0.5, verbose=False)

        # Process results
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # Get bounding box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])

                # Draw bounding box around the face
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label text
                label = f"Face: {confidence:.2f}"
                cv2.putText(
                    frame, 
                    label, 
                    (x1, max(y1 - 10, 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                )

        # Display the resulting frame
        cv2.imshow('YOLOv8 Face Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()