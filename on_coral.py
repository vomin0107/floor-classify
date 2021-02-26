from edgetpu.classification.engine import ClassificationEngine
from PIL import Image
import time
import cv2
import re
import os

# the TFLite converted to be used with edgetpu
modelPath = 'floor_type_classify_models/converted_edgetpu/model_edgetpu.tflite'

# The path to labels.txt that was downloaded with your model
labelPath = 'floor_type_classify_models/converted_edgetpu/labels.txt'

# This function parses the labels.txt and puts it in a python dictionary
def loadLabels(labelPath):
    p = re.compile(r'\s*(\d+)(.+)')
    with open(labelPath, 'r', encoding='utf-8') as labelFile:
        lines = (p.match(line).groups() for line in labelFile.readlines())
        return {int(num): text.strip() for num, text in lines}

# This function takes in a PIL Image from any source or path you choose
def classifyImage(image_path, engine):
    # Load and format your image for use with TM2 model
    # image is reformated to a square to match training
    #image = Image.open(image_path)
    #image.resize((224, 224))

    # Classify and ouptut inference
    classifications = engine.classify_with_image(image_path)
    return classifications

def main():
    # Load your model onto your Coral Edgetpu
    engine = ClassificationEngine(modelPath)
    labels = loadLabels(labelPath)

    cap = cv2.VideoCapture(0)

    prev_time = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        cur_time = time.time()
        sec = cur_time - prev_time
        prev_time = cur_time
        fps = str(round(1 / sec, 1)) + 'fps '

        # Format the image into a PIL Image so its compatable with Edge TPU
        cv2_im = frame
        cv2_im_input = cv2.resize(frame, (224, 224))
        pil_im = Image.fromarray(cv2_im_input)

        # Resize and flip image so its a square and matches training
        #pil_im.resize((224, 224))
        pil_im.transpose(Image.FLIP_LEFT_RIGHT)

        # Classify and display image
        results = classifyImage(pil_im, engine)
        result[0][0] = class_value
        probability = str(round(results[0][1]*100,1)) + '%'
        if class_value == 0:
            result = 'Wood '
        elif class_value == 1:
            result = 'Yellow '
        elif class_value == 2:
            result = 'Marble '
        else:
            result = 'Carpet '
        result = fps + result + probability
        cv2.putText(frame, result, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0))
        cv2.imshow('frame', cv2_im)
        # print(results)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()