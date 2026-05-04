import cv2
from ultralytics import YOLO


model = YOLO("yolov8n.pt") 


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    
    
    results = model(img, stream=True)

    for r in results:
        boxes = r.boxes
        for box in boxes:
        
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
    
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

        
            conf = round(float(box.conf[0]), 2)

    
            cls = int(box.cls[0])
            label = model.names[cls]

    
            cv2.putText(img, f"{label} {conf}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


    cv2.imshow("Object Detection", img)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
