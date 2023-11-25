import os
import random
from PIL import Image
import cv2
from ultralytics import YOLO
import time
# from tracker import Tracker

# video_path = os.path.join( './Data', 'pedestrian.mp4')
# video_out_path = os.path.join('.', 'pedestrianDetected.mp4')


video_path = os.path.join( './Data', 'project_video.mp4')
video_out_path = os.path.join('.', 'project_videoDetected.mp4')

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
print(frame)
cap_out = cv2.VideoWriter(video_out_path, cv2.VideoWriter_fourcc(*'MP4V'), cap.get(cv2.CAP_PROP_FPS),(frame.shape[1], frame.shape[0]))



model = YOLO("yolov8n.pt")
dt_thr=0.3
while ret:
    t1 = time.time()
    results =  model.predict(frame)
    for result in results:
        # print(result) 
        detections = []
        for idx,r in enumerate(result.boxes.data.tolist()):
            x1, y1, x2, y2, score, class_id = r
            # if dt_thr<float(result.boxes.conf[idx]):
            #     break
            x1 = int(x1)
            x2 = int(x2)
            y1 = int(y1)
            y2 = int(y2)
            class_id = int(class_id)
            cv2.putText(frame, str(result.names[int(result.boxes.cls[idx])]), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.putText(frame, str(result.names[int(result.boxes.cls[idx])]), (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 3)
    fps = 1./(time.time()-t1)
    cv2.putText(frame, "FPS: {:.2f}".format(fps), (0,30), 0, 1, (0,0,255), 2)
    cv2.imshow('Object Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    cap_out.write(frame)
    ret, frame = cap.read()
    
#     # ret, frame = cap.read()

cap.release()
cap_out.release()
cv2.destroyAllWindows()