import numpy as np
import cv2
import keyboard

# face detection
face_cascade_url = "./haarcascades/haarcascade_frontalface_alt.xml"
eye_cascade_url = "./haarcascades/haarcascade_mcs_eyepair_big.xml"
nose_cascade_url = "./haarcascades/haarcascade_mcs_nose.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_url)
eyes_cascade = cv2.CascadeClassifier(eye_cascade_url)
nose_cascade = cv2.CascadeClassifier(nose_cascade_url)
glasses = cv2.imread("Images/glasses.png", -1)
mustache = cv2.imread("Images/mustache.png", -1)


def filter_on_jamie():
    img = cv2.imread("./Images/Jamie_Before.jpg")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + h]  # rec
        roi_color = img[y:y + h, x:x + h]
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)

        eyes = eyes_cascade.detectMultiScale(roi_gray, 1.5, 5)
        for (ex, ey, ew, eh) in eyes:
            # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)
            roi_eyes = roi_gray[ey: ey + eh, ex: ex + ew]
            glasses2 = cv2.resize(glasses.copy(), (int(ew), int(2 * eh)))

            gw, gh, gc = glasses2.shape
            for i in range(0, gw):
                for j in range(0, gh):
                    # print(glasses[i, j]) #RGBA
                    if glasses2[i, j][3] != 0:  # alpha 0
                        roi_color[ey - int(eh / 2) + i, ex + j] = glasses2[i, j]

        nose = nose_cascade.detectMultiScale(roi_gray, 1.5, 5)
        for (nx, ny, nw, nh) in nose:
            # cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 3)
            roi_nose = roi_gray[ny: ny + nh, nx: nx + nw]
            mustache2 = cv2.resize(mustache.copy(), (nw, int(0.5 * ny)))

            mw, mh, mc = mustache2.shape
            for i in range(0, mw):
                for j in range(0, mh):
                    # print(glasses[i, j]) #RGBA
                    if mustache2[i, j][3] != 0:  # alpha 0
                        roi_color[ny + int(nh / 2.0) + i, nx + j] = mustache2[i, j]
        cv2.imwrite("Images/Jamie_After.jpg", img)
    cv2.imshow("Jamie", img)
    cv2.waitKey(0)

    # Display the resulting frame
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


def filter_on_picture2():
    img = cv2.imread("./Images/Before.png")
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.5, 5)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + h]  # rec
        roi_color = img[y:y + h, x:x + h]
        # cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 3)

        eyes = eyes_cascade.detectMultiScale(roi_gray, 1.5, 5)
        for (ex, ey, ew, eh) in eyes:
            # cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 3)
            roi_eyes = roi_gray[ey: ey + eh, ex: ex + ew]
            glasses2 = cv2.resize(glasses.copy(), (int(ew), int(2.6 * eh)))

            gw, gh, gc = glasses2.shape
            for i in range(0, gw):
                for j in range(0, gh):
                    # print(glasses[i, j]) #RGBA
                    if glasses2[i, j][3] != 0:  # alpha 0
                        roi_color[ey - int(eh / 2) + i, ex + j] = glasses2[i, j]

        nose = nose_cascade.detectMultiScale(roi_gray, 1.5, 5)
        for (nx, ny, nw, nh) in nose:
            # cv2.rectangle(roi_color, (nx, ny), (nx + nw, ny + nh), (255, 0, 0), 3)
            roi_nose = roi_gray[ny: ny + nh, nx: nx + nw]
            mustache2 = cv2.resize(mustache.copy(), (nw, int(0.5 * ny)))

            mw, mh, mc = mustache2.shape
            for i in range(0, mw):
                for j in range(0, mh):
                    # print(glasses[i, j]) #RGBA
                    if mustache2[i, j][3] != 0:  # alpha 0
                        roi_color[ny + int(nh / 1.5) + i, nx + int(nw / 5.0) - 2 + j] = mustache2[i, j]
        cv2.imwrite("Images/After.jpg", img)
    cv2.imshow("Got Character", img)
    cv2.waitKey(0)

    # Display the resulting frame
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


option = input('''Enter your option -
                    1 . Filter on picture 1 - Jamie's Picture
                    2 . Filter on picture 2 
                    Enter your choice here - ''')
if option == "1":
    filter_on_jamie()
elif option == "2":
    filter_on_picture2()
else:
    print("Enter valid option")

# while True:
#
#
#
#
#     key_pressed = cv2.waitKey(1) & 0xFF
#     if key_pressed == ord('q'):
#         break
#     if keyboard.is_pressed('q'):
#         break
cv2.destroyAllWindows()