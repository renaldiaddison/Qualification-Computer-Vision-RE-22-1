import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import os

video = cv.VideoCapture(0)
classifier = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

face_list = []
class_list = []
names = []

train_path = 'images/train'
tdir = os.listdir(train_path)

for index, train_dir in enumerate(tdir):
    names.append(train_dir)
    for image_path in os.listdir(f'{train_path}/{train_dir}'):
        path = f'{train_path}/{train_dir}/{image_path}'
        if path.split('.')[1] != 'db':
            gray = cv.imread(path, 0)
            faces = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
            if len(faces) < 1:
                continue
            for face_rect in faces:
                x, y, w, h = face_rect
                face_image = gray[y: y + w, x : x + h]
                face_list.append(face_image)
                class_list.append(index)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(face_list, np.array(class_list))

def stackImages(scale,img_array):
    rows = len(img_array)
    cols = len(img_array[0])
    available_rows = isinstance(img_array[0], list)
    width = img_array[0][0].shape[1]
    height = img_array[0][0].shape[0]
    if available_rows:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if img_array[x][y].shape[:2] == img_array[0][0].shape [:2]:
                    img_array[x][y] = cv.resize(img_array[x][y], (0, 0), None, scale, scale)
                else:
                    img_array[x][y] = cv.resize(img_array[x][y], (img_array[0][0].shape[1], img_array[0][0].shape[0]), None, scale, scale)
                if len(img_array[x][y].shape) == 2: img_array[x][y]= cv.cvtColor(img_array[x][y], cv.COLOR_GRAY2BGR)
        blank_image = np.zeros((height, width, 3), np.uint8)
        horiz = [blank_image]*rows
        for x in range(0, rows):
            horiz[x] = np.hstack(img_array[x])
        vertic = np.vstack(horiz)
    else:
        for x in range(0, rows):
            if img_array[x].shape[:2] == img_array[0].shape[:2]:
                img_array[x] = cv.resize(img_array[x], (0, 0), None, scale, scale)
            else:
                img_array[x] = cv.resize(img_array[x], (img_array[0].shape[1], img_array[0].shape[0]), None,scale, scale)
            if len(img_array[x].shape) == 2: img_array[x] = cv.cvtColor(img_array[x], cv.COLOR_GRAY2BGR)
        horiz = np.hstack(img_array)
        vertic = horiz
    return vertic

def blur_image(frame_blur_copy):
    gray_blur = cv.cvtColor(frame_blur_copy, cv.COLOR_BGR2GRAY)
    faces_blur = classifier.detectMultiScale(gray_blur, scaleFactor=1.2, minNeighbors=5)

    for(x, y, w, h) in faces_blur:

        image2 = cv.rectangle(frame_blur_copy, (x,y), (x + w, y + h), (0, 255, 0), 2)
        face_image2 = gray_blur[y: y + w, x: x + h]
        idx2, similarity2 = face_recognizer.predict(face_image2)

        if(similarity2 > 100.0):
            similarity2 = 100.0

        # Putting text name and similarity percent
        text = f'{names[idx2]} {(int(similarity2))}%'
        cv.putText(frame_blur_copy, text, (x, y-20), cv.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 2)

        image2[y : y + h, x : x + w] = cv.blur(image2[y : y + h, x : x + w], (50,50))

    cv.putText(frame_blur_copy, "Blur Filter", (10, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    return frame_blur_copy

def cequ_gray(frame_cequ):
    cv.putText(frame_cequ, "Cequ Gray Filter", (10, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    frame_gray = cv.cvtColor(frame_cequ, cv.COLOR_BGR2GRAY)
    clahe = cv.createCLAHE(clipLimit=4.0, tileGridSize = (8, 8))
    cequ_gray = clahe.apply(frame_gray)

    return cequ_gray

def face_processing(frame):
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = classifier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
    similarity = 0

    for(x, y, w, h) in faces:
        image = cv.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)
        face_image = gray[y: y + w, x: x + h]
        idx, similarity = face_recognizer.predict(face_image)

        if(similarity > 100.0):
            similarity = 100.0

        # Putting text name and similarity percent
        text = f'{names[idx]} {(int(similarity))}%'
        cv.putText(frame, text, (x, y-20), cv.FONT_HERSHEY_PLAIN, 1.3, (0, 255, 0), 2)

    cv.putText(frame, "Face Detection", (10, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    return frame

def shape_detector(frame_s):
    frame_detector_gray = cv.cvtColor(frame_s, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(frame_detector_gray, 110, 255, cv.THRESH_BINARY)
    _, contours, _ = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        approx = cv.approxPolyDP(contour, 0.01 * cv.arcLength(contour, True), True)
        cv.drawContours(frame_s, [contour], 0, (0, 0, 0), 2)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5

        if len(approx) == 3:
            cv.putText(frame_s, 'Triangle', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        elif len(approx) == 4:
            x, y, w, h = cv.boundingRect(approx)
            aspect_ratio = float (w) / h
            if aspect_ratio >= 0.95 and aspect_ratio <= 1.05:
                cv.putText(frame_s, 'Square', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

            else:
                cv.putText(frame_s, 'Rectangle', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        elif len(approx) == 5:
            cv.putText(frame_s, 'Pentagon', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        elif len(approx) == 6:
            cv.putText(frame_s, 'Hexagon', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        elif len(approx) == 10:
            cv.putText(frame_s, 'Star', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        else:
            cv.putText(frame_s, 'Circle', (x, y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    cv.putText(frame_s, "Shape Detection", (10, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    return frame_s

def canny_images(frame_canny):
    cv.putText(frame_canny, "Canny Filter", (10, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)
    gray_canny = cv.cvtColor(frame_canny, cv.COLOR_BGR2GRAY)
    canny_frame = cv.Canny(gray_canny, 240, 90)

    return canny_frame

def original_image(frame_original):
    cv.putText(frame_original, "Original", (10, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

    return frame_original

while True:

    # frame with face recognition
    _, frame = video.read()

    # no edited frame
    no_edited_frame = frame.copy()

    # cequ gray frame
    frame_cequ = frame.copy()

    # shape detector frame
    frame_shape = frame.copy()

    # blur frame
    frame_blur_copy = frame.copy()

    # canny frame
    frame_canny_copy = frame.copy()


    

    #stack the images
    img_stack = stackImages(0.7, [[original_image(no_edited_frame), face_processing(frame), shape_detector(frame_shape)], [blur_image(frame_blur_copy), canny_images(frame_canny_copy), cequ_gray(frame_cequ)],])

    cv.imshow('Qualification RE22-1',img_stack)
    key = cv.waitKey(1)

    #press escape to quit
    if (key==27):
        path = 'images/result'
        filename = 'result.jpg'
        cv.imwrite(os.path.join(path, filename), img_stack)
        saved_image = cv.imread(path + '/' + filename, cv.IMREAD_ANYCOLOR)
        break

video.release()
cv.destroyAllWindows()