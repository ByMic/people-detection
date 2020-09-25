import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
 
def detectPersons(img_path, save_dir, weight_path, config_path, classes_file, conf_thresh,nms_thresh,whT):
    
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    base = os.path.basename(img_path)
    img_name = os.path.splitext(base)[0]
    
    class_names = []
    with open(classes_file, 'rt') as f:
        class_names = f.read().rstrip('\n').split('\n')

    net = cv.dnn.readNetFromDarknet(config_path, weight_path)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)
   
    img = plt.imread(img_path)
    blob = cv.dnn.blobFromImage(img, 1.0/255, (whT, whT), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layers_names = net.getLayerNames()
    output_names = [(layers_names[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_names)

    hT, wT, cT = img.shape
    summary = {}
    summary["file"] = img_name + ".jpg"
    summary["width"] = wT
    summary["height"] = hT

    bbox = []
    class_ids = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > conf_thresh:
                w,h = int(det[2]*wT) , int(det[3]*hT)
                x,y = int((det[0]*wT)-w/2) , int((det[1]*hT)-h/2)
                bbox.append([x,y,w,h])
                class_ids.append(class_id)
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, conf_thresh, nms_thresh)

    cnt = 1
    objects = []
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        if class_ids[i]==0:
            object_data = {}
            object_data["file"] = img_name + "_" + str(cnt) + ".jpg"
            object_data["top"] = y
            object_data["left"] = x
            object_data["bottom"] = y+h
            object_data["right"] = x+w
            crop_img = img[y:y+h, x:x+w]
            plt.imsave(os.path.join(save_dir,object_data["file"]),crop_img)
            objects.append(object_data)
            cnt += 1

            cv.rectangle(img, (x, y), (x+w,y+h), (255, 0 , 255), 2)
            cv.putText(img,f'{class_names[class_ids[i]].upper()} {int(confs[i]*100)}%',
                  (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

    summary["num_person"] = cnt-1
    summary["objects"] = objects

    with open(os.path.join(save_dir,img_name+".json"), 'w') as outfile:
        json.dump(summary, outfile, indent=2)

    plt.imshow(img)
    plt.show()