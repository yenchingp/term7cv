from ultralytics import YOLO
from PIL import Image


def detect_objects(img_path: str, output_path: str):
    # Load a model
    model = YOLO('Models/yolov8_best.pt')  # pretrained YOLOv8n model

    # Run batched inference on a list of images
    results = model(source=img_path)

    i=0 # keep track of output
    for r in results:
        im_array = r.plot(labels=False)  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        # im.show()  # show image
        file_name = f"output_{i}.jpg"
        im.save(file_name)
        boxes = results[i].boxes.xyxy.tolist()
        for box in boxes:
            print(box) #(x1,y1,x2,y2)
            i+=1

