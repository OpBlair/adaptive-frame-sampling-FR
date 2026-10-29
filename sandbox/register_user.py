import os
import cv2


def register_user():
  # 1. Setup user details and directories
  user_name = input("Enter your name for registration: ").strip()
  if not user_name:
    print("[ERROR] Name cannot be empty.")
    return

  dataset_dir = "dataset"
  user_path = os.path.join(dataset_dir, user_name)
  os.makedirs(user_path, exist_ok=True)

  # 2. Initialize webcam and Haar Cascade face detector
  cap = cv2.VideoCapture(0)
  if not cap.isOpened():
    print("[ERROR] Could not open webcam.")
    return

  # Load OpenCV's pre-trained face detection cascade
  face_cascade = cv2.CascadeClassifier(
      cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
  )

  print(f"\n[INFO] Initializing registration for: {user_name}")
  print(
      "[INSTRUCTIONS] Position your face in front of the camera. Press 'c'"
      " to capture a sample."
  )
  print(
      "[INSTRUCTIONS] Try shifting angles slightly (tilt head, look left/right)"
      " for robust data."
  )
  print("[INSTRUCTIONS] Target: ~20-30 samples. Press 'q' to quit early.\n")

  count = 0
  target_samples = 30

  while True:
    ret, frame = cap.read()
    if not ret:
      print("[ERROR] Failed to grab video frame.")
      break

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5, minSize=(50, 50)
    )

    # Draw bounding box around detected faces
    for x, y, w, h in faces:
      cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display status text on the live video feed
    status_text = f"Captured: {count}/{target_samples} | Press 'c' to snap, 'q' to quit"
    cv2.putText(
        frame,
        status_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
    )

    cv2.imshow("User Registration Tool", frame)

    key = cv2.waitKey(1) & 0xFF

    # Press 'c' to capture current face crop
    if key == ord("c"):
      if len(faces) > 0:
        # Grab the first detected face
        x, y, w, h = faces[0]
        face_roi = frame[y : y + h, x : x + w]

        count += 1
        img_path = os.path.join(user_path, f"{user_name}_{count}.jpg")
        cv2.imwrite(img_path, face_roi)
        print(f"[SUCCESS] Saved sample {count}/{target_samples} -> {img_path}")
      else:
        print("[WARNING] No face detected in frame! Adjust position.")

    # Quit if 'q' is pressed or target count is reached
    elif key == ord("q") or count >= target_samples:
      break

  # Cleanup
  cap.release()
  cv2.destroyAllWindows()
  print(
      f"\n[INFO] Registration finished! Saved {count} face samples for"
      f" '{user_name}' in ./{user_path}/"
  )


if __name__ == "__main__":
  register_user()