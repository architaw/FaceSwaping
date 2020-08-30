import cv2

cap = cv2.VideoCapture(0)

classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

while True:
    retval, image = cap.read()  # retval is return value

    if retval:

        faces = classifier.detectMultiScale(image)

        if len(faces) >=2: #faces should be 2 or more for swapping

            sorted_faces=sorted(faces, key=lambda item:item[2]*item[3]) #sorted acc to area i.e w*h

            face1,face2=sorted_faces[-2:] #last two faces largest and second largest

            x1, y1, w1, h1 = face1 #last face will be largest face

            x2, y2, w2, h2 = face2

            cut1 = image[y1:y1+h1,x1:x1+w1]
            cut2 = image[y2:y2 + h2, x2:x2 + w2]

            t_cut1 = cv2.resize(cut2, (cut1.shape[1], cut1.shape[0]))
            t_cut2 = cv2.resize(cut1, (cut2.shape[1], cut2.shape[0]))

            image[y1:y1 + h1, x1:x1 + w1] = t_cut1
            image[y2:y2 + h2, x2:x2 + w2] = t_cut2

            cv2.imshow("swapped", image)
        else:
            cv2.imshow("swapped", image)

    key=cv2.waitKey(1)  # 1ms wait

    if key == ord("q"): #ord gives ascii value of any char
        break
    elif key==ord("c"):
        cv2.imwrite("classroom.jpg",image)

cap.release()
cv2.destroyAllWindows()
