import cv2
from ultralytics import YOLO

# Load YOLO model (pre-trained)
model = YOLO("yolov8n.pt")   # small & fast model

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    
    # Run detection
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # Draw rectangle
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

            # Confidence
            conf = round(float(box.conf[0]), 2)

            # Class name
            cls = int(box.cls[0])
            label = model.names[cls]

            # Put text
            cv2.putText(img, f"{label} {conf}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Show output
    cv2.imshow("Object Detection", img)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
