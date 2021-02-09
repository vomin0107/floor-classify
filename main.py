import cv2
import tensorflow.keras
import numpy as np
import time

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = tensorflow.keras.models.load_model('keras_model.h5', compile = False)
label_file = open('labels.txt', 'r')
label = []

for l in label_file:
    if l != '':
        label.append(l[2:-1])
# print(label)

data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

size = (224, 224)

################# opencv camera ##################

# print(cv2.__version__)

cap = cv2.VideoCapture(0)

print('width :%d, height : %d' % (cap.get(3), cap.get(4)))

prev_time = 0
while True:
    ret, frame = cap.read()    # Read 결과와 frame

    cur_time = time.time()
    sec = cur_time-prev_time
    prev_time = cur_time
    fps = str(round(1/sec, 1))

    # image = ImageOps.fit(frame, size, Image.ANTIALIAS)
    image = cv2.resize(frame, size)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    maxi = max(prediction[0])

    fps = fps + 'fps ' + label[prediction[0].tolist().index(maxi)] + ' ' + str(round(maxi * 100, 1)) + '%'

    if(ret) :
        cv2.putText(frame, fps, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0))
        cv2.imshow('frame_color', frame)    # 컬러 화면 출력
        if cv2.waitKey(1) == 27:
            break
cap.release()
cv2.destroyAllWindows()