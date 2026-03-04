import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    faces = face_cascade.detectMultiScale(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.3, 5
    )

    for face in faces:
        cv2.rectangle(
            frame,
            (face[0], face[1]),
            (face[0] + face[2], face[1] + face[3]),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Age: 20-30",
            (face[0], face[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()