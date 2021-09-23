import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3.weights','yolov3.cfg')
classes = []
with open('coco.names','r') as f:
    classes = [line.strip() for line in f.readlines()]

#get the names of the unused layers of the YOLO dnn
outputlayers = net.getUnconnectedOutLayersNames()
img = cv2.imread('workplace.jpg')
img = cv2.resize(img, None, fx=0.5, fy=0.5)

#obtain the blobs from the image
blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), 
                             (0,0,0), True, crop=False)

net.setInput(blob)
outs = net.forward(outputlayers)

h, w = img.shape[:2]
confidences = []
classIDs = []
boxes = []
for out in outs:
    for detection in out:
        
        scores = detection[5:]
        
        #obtain the classid for the objects detected in blob with max score
        classID = np.argmax(scores)
        

        confidence = scores[classID]
        if confidence > 0.5:
            box = detection[:4] * np.array([w, h, w, h])
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            box = [x, y, int(width), int(height)]
            boxes.append(box)
            confidences.append(float(confidence))            
            classIDs.append(classID)

indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
if len(indices) > 0:
    for i in indices.flatten():
        label = classes[classIDs[i]]
        (x, y) = (boxes[i][0], boxes[i][1])
        (w, h) = (boxes[i][2], boxes[i][3])
        cv2.rectangle(img, (x,y),(x+w, y+h),(0,0,255),2)
        cv2.putText(img, str(label),(x, y-20), cv2.FONT_HERSHEY_PLAIN, 1,(0,255,0), 2)
cv2.imshow('FINAL', img)
cv2.waitKey(1)
cv2.destroyAllWindows()