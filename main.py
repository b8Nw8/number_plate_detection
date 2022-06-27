import tensorflow as tf
import keras
import cv2
import numpy as np
import gdown
import sys
from generate_data import decode_batch

carPlatesCascade = cv2.CascadeClassifier('haarcascade_russian_plate_number.xml')

model = keras.models.load_model('ocr_model.h5')

abc = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y']
digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8' '9']
letters = ['A', 'B', 'C', 'E', 'H', 'K', 'M', 'O', 'P', 'T', 'X', 'Y']

output = 'cache.mp4'
url = sys.argv[1]
gdown.download(url=url, output=output, quiet=False, fuzzy=True)


# def get_data(plate, counter):
#     img_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
#     se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
#     bg = cv2.morphologyEx(img_gray, cv2.MORPH_DILATE, se)
#     out_gray = cv2.divide(img_gray, bg, scale=255)
#     out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]
#     image = cv2.resize(out_binary, (128, 64))
#     imageT = image.T
#     image_array = imageT.astype(np.float32)
#     image_array /= 255
#     X_data[counter, :, :] = image_array
#     counter += 1
#     return counter


cap = cv2.VideoCapture('cache.mp4')
# cap.set(cv2.CAP_PROP_FPS, 2)

if not cap.isOpened():
    print('Error Reading video')

while cap.isOpened:
    ret, frame = cap.read()
    if ret is True:

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        car_plates = carPlatesCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, maxSize=(150, 300))

        output = frame.copy()
        counter = 0
        #box = []
        #topop = []

        check = type(car_plates) is tuple
        if not check:
            X_data = np.zeros((car_plates.shape[0], 128, 64))
        else:
            X_data = np.zeros((1, 128, 64))

        for x, y, w, h in car_plates:
            plate = frame[y: y + h, x: x + w]
            img_gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
            se = cv2.getStructuringElement(cv2.MORPH_RECT, (8, 8))
            bg = cv2.morphologyEx(img_gray, cv2.MORPH_DILATE, se)
            out_gray = cv2.divide(img_gray, bg, scale=255)
            out_binary = cv2.threshold(out_gray, 0, 255, cv2.THRESH_OTSU)[1]
            image = cv2.resize(out_binary, (128, 64))
            imageT = image.T
            image_array = imageT.astype(np.float32)
            image_array /= 255
            #X_data[counter, :, :] = image_array
            batch = np.expand_dims(image_array, axis=-1)
            batch = np.expand_dims(batch, axis = 0)
            get_tensor = tf.convert_to_tensor(batch)
            net_out_value = model.predict(get_tensor)
            pred_texts = decode_batch(net_out_value)
            shapes = np.zeros_like(frame, np.uint8)
            print(x,y,w,h)
            if y > h + 15:
                if x + w + 15 < frame.shape[1]:
                    shapes[y - h - 15: y - 15, x + 15: x + w + 15] = frame[y: y + h, x: x + w]
                else:
                    shapes[y - h - 15: y - 15, x - w - 15: x - 15] = frame[y: y + h, x: x + w]
            else:
                if x + w + 15 < frame.shape[1]:
                    shapes[y + 15: y + h + 15, x + 15: x + w + 15] = frame[y: y + h, x: x + w]
                else:
                    shapes[y + 15: y + h + 15, x - w - 15: x - 15] = frame[y: y + h, x: x + w]
            shapes_gray = cv2.cvtColor(shapes, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(shapes_gray, (3, 3), 0)
            edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
            mask = shapes_gray.astype(bool)
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.rectangle(output, (x + 15, y - 15), (x + w + 15, y - h - 15), (0, 0, 255), 2)
            output[mask] = cv2.addWeighted(output, 0, shapes,
                                           1, 0)[mask]
            cv2.putText(output, '%s' % pred_texts[0], (x + w + 30, y - 20),
                        cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
            cv2.imshow('Video', output)

        # topop = []
        #
        # for i in range(len(box)):
        #     if len(pred_texts[i]) > 9 or len(pred_texts[i]) < 8 or pred_texts[i][0] in digits or \
        #                                      pred_texts[i][4] in digits or pred_texts[i][5] in digits or \
        #                                      pred_texts[i][1] in letters or pred_texts[i][2] in letters or \
        #                                      pred_texts[i][3] in letters or pred_texts[i][6] in letters or \
        #                                      pred_texts[i][7] in letters:
        #         topop.append(box[i])
        #
        # for i in range(len(topop)):
        #     box.pop(i)
        #     pred_texts.pop(i)
        #     print(pred_texts[i])

        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    else:
        break

cap.release()
cv2.destroyAllWindows()
