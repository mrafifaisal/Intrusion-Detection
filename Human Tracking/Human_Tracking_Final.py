# TechVidvan Vehicle counting and Classification

# Import necessary packages

import cv2
from csv import writer
import collections
import numpy as np
from tracker import *
# from sort import *

# Initialize Tracker
tracker = EuclideanDistTracker()
# tracker = Sort()
# Initialize the videocapture object
cap = cv2.VideoCapture('Media/VIRAT_S_010204_05_000856_000890.mp4')
# cap = cv2.VideoCapture('I:/OneDrive - Higher Education Commission/NUST Funded Projects/NCRA/Computer Vision Project/Research Video/DJI_0753.MP4')

input_size = 320
points_dict ={}
global img
# success, cape = cap.read()
CNTR = 0
# Detection confidence threshold
confThreshold =0.50
nmsThreshold= 0.2
roi_flag=False
font_color = (0, 0, 255)
font_size = 0.5
font_thickness = 2
Frame_Number = 1
ctr_pnts = []

# Store Coco Names in a list
classesFile = "coco.names"
classNames = open(classesFile).read().strip().split('\n')
# print(classNames)
#print(len(classNames))

# class index for our required detection classes
#required_class_index = [classNames.index(value) for value in classNames]
required_class_index = [0]
# print(required_class_index)
detected_classNames = []
boxes_ids = []
## Model Files
modelConfiguration = 'yolov3-320.cfg'
modelWeigheights = 'yolov3-320.weights'

# width = 480
# height = 640
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
# codec = cv2.VideoWriter_fourcc(*FLAGS.output_format)
# out = cv2.VideoWriter(FLAGS.output, codec, fps, (width, height))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
size = (width, height)
fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
out = cv2.VideoWriter(r'I:\OneDrive - Higher Education Commission\NUST Funded Projects\NCRA\Computer Vision Project\vehicle-detection-classification-opencv\outputs\out_1.avi',fourcc, fps, size,True)

# configure the network model
net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeigheights)

#Configure the network backend
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
#net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

# Define random colour for each class
np.random.seed(42)
colors = np.random.randint(0, 255, size=(len(classNames), 3), dtype='uint8')

def add_to_dict(d,key,val):
    if key not in d:
        d[key]=[val]
    elif type(d[key])==list:
        d[key].append(val)
    else:
        d[key]=[d[key],val]
# Function for finding the center of a rectangle
def find_center(x, y, w, h):
    x1=int(w/2)
    y1=int(h/2)
    cx = x+x1
    cy=y+y1
    return cx, cy

#Function for csv data
def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)
    write_obj.close()
    
def Intrusion(centre):
    try:
        if (centre[0] > refPt[0][0]) and (centre[0] < refPt[1][0]) and (centre[1] > refPt[0][1]) and (centre[1] < refPt[1][1]):
            return True
        else:
            return False
    except:
        print('No Intrusion')

# Function for finding the detected objects from the network output
def postProcess(outputs,img):
    global detected_classNames 
    height, width = img.shape[:2]
    boxes = []
    classIds = []
    confidence_scores = []
    detection = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if classId in required_class_index:
                if confidence > confThreshold:
                    # print(classId)
                    w,h = int(det[2]*width) , int(det[3]*height)
                    x,y = int((det[0]*width)-w/2) , int((det[1]*height)-h/2)
                    boxes.append([x,y,w,h])
                    classIds.append(classId)
                    confidence_scores.append(float(confidence))

    # Apply Non-Max Suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidence_scores, confThreshold, nmsThreshold)
    # print(classIds)
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            # print(x,y,w,h)
    
            color = [int(c) for c in colors[classIds[i]]]
            name = classNames[classIds[i]]
            detected_classNames.append(name)
            # Draw classname and confidence score 
            # cv2.putText(img,f'{name.upper()} {int(confidence_scores[i]*100)}%',
            #           (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.putText(img,f'{int(confidence_scores[i]*100)}%',
                      (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
            # Draw bounding rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 1)
            # detection.append([x, y, w, h, int(confidence_scores[i]*100)])
            detection.append([x, y, w, h, required_class_index.index(classIds[i])])
            # print("Detections", detection)
    # Update the tracker for each object
    boxes_ids = tracker.update(detection)
    cv2.putText(img, "Objects being tracked: {}".format(len(boxes_ids)), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)

    # print("Box_ID", boxes_ids)
    for box_id in boxes_ids:
        x, y, w, h, id, index = box_id
        print(len(box_id))

        # Find the center of the rectangle for detection
        center = find_center(x, y, w, h)
        ix, iy = center
        ctr_pnts.append(center)
       
        # Draw circle in the middle of the rectangle
        cv2.circle(img, (ix, iy), 2, (0, 0, 255), -1)
        cv2.putText(img, str(id), (x+10, y - 30), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)
        
        #print(id, x, y, w, h, ix, iy )
        Traj = [id, x, y, w, h, ix, iy]
        add_to_dict(points_dict,id,center)
        CNTR = center
        print('Total ID: ',len(points_dict))
        for key in points_dict:
           
            for item in (points_dict[id]):
                # print("all:", item)
                cv2.circle(img, (item), 2, (255, id*30, id*30), -1)
                # print('CNTR: ', CNTR)
                signal = Intrusion(CNTR)
                if signal == True:
                    cv2.putText(img, "Intrusion Detected!!", (5, 120), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (255, 0, 0), 2)
                # print(points_dict[key]) 
        cv2.putText(img, "Total ID being tracked: {}".format(len(points_dict)), (5, 80), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 255, 0), 2)
        append_list_as_row('Trajectory11.csv', Traj)
    
refPt = []
cropping = False

def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global refPt, cropping

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		refPt = [(x, y)]
		cropping = True

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		refPt.append((x, y))
		cropping = False
    
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)

def realTime():
    while True:
        success, img = cap.read()
        # img = cv2.resize(img,(640,480))
        ih, iw, channels = img.shape
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (input_size, input_size), [0, 0, 0], 1, crop=False)
        # List of strings
        # print('==========+======+==========')

        global Frame_Number
        NoFrames = ['Frame_Number', Frame_Number]
        Frame_Number += 1
        # Append a list as new line to an old csv file
        append_list_as_row('Trajectory.csv', NoFrames)
        # Set the input of the network
        net.setInput(blob)
        layersNames = net.getLayerNames()
        outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
        # Feed data to the network
        outputs = net.forward(outputNames)
    
        # Find the objects from the network output
        postProcess(outputs,img)
        try:
            cv2.rectangle(img, refPt[0], refPt[1], (0, 255, 0), 2)
            print("ROI: ", refPt[0], refPt[1])
            # print('center in try',CNTR)
            # Intrusion(CNTR)
        except:
            print("No ROI")
            
        
    
        result = np.asarray(img)
        out.write(result)
        #Show the frames
        cv2.imshow('image', img)
        Press = cv2.waitKey(1) & 0xFF
        if Press == ord('r'):
            refPt[0] = refPt[1] = 0
            
        elif Press == ord('q'):
            break

    # Finally realese the capture object and destroy all active windows
    cap.release()
    # print(points_dict)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    Heading = ['ID', 'X', 'Y', 'W', 'H', 'iX', 'iY']
    append_list_as_row('Trajectory.csv', Heading)
    realTime()
