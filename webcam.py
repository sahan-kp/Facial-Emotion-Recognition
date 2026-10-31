import cv2
from src.predict_utils import load_model, predict_face

def main():
    model = load_model()
    detector = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Webcam could not be opened. Check camera permission.")

    print("Press Q to quit.")
    while True:
        ok, frame = camera.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
        )

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            label, confidence, _ = predict_face(model, face)
            text = f"{label} ({confidence * 100:.1f}%)"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (40, 220, 100), 2)
            cv2.putText(frame, text, (x, max(y-10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 220, 100), 2)

        cv2.imshow("AI Facial Expression Recognition - Press Q to quit", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
