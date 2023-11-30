import os
import csv
from ultralytics import YOLO
from PIL import Image


def detect_objects(img_path: str, img_name: str, output_path: str):
    # Load a model
    model = YOLO('Models/yolov8_best.pt')  # pretrained YOLOv8n model

    # Run batched inference on a list of images
    results = model(source=img_path)

    i=0 # keep track of output
    for r in results:
        im_array = r.plot(labels=False)  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image

        annotated_img = f'{output_path}/{img_name.split(".")[0]}_annotated.jpg'
        im.save(annotated_img)

        annotations_file = f'{output_path}/{img_name.split(".")[0]}_annotations.csv'
        with open(annotations_file, 'w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            boxes = results[i].boxes.xyxy.tolist()
            for box in boxes:
                writer.writerow(box)
                i+=1

    return True


def extract_objects(img_path: str, img_name: str, output_path: str):
    raw_img = Image.open(img_path)
    
    obj_output_path = f'{output_path}/{img_name.split(".")[0]}_objects'
    if not os.path.exists(obj_output_path):
        os.mkdir(obj_output_path)

    annotations_file = f'{output_path}/{img_name.split(".")[0]}_annotations.csv'
    with open(annotations_file, 'r') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
            obj = raw_img.crop((x1, y1, x2, y2))
            obj.save(f'{obj_output_path}/obj_{idx}.jpg')

    return True

