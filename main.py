import cv2

#loading test image
img = cv2.imread('faces2.jpg')

#convert the image to grayscale
image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#initialize the face detector using haar cascade

face_cascade = cv2.CascadeClassifier("cascades\haarcascade_frontalface_default.xml")

# detect all the faces in the image
faces = face_cascade.detectMultiScale(image_gray)
# print the number of faces detected
print(f"{len(faces)} faces detected in the image.")

# for every face, draw a blue rectangle
for x, y, width, height in faces:
    cv2.rectangle(img, (x, y), (x + width, y + height), color=(255, 0, 0), thickness=2)

# save the image with rectangles
cv2.imwrite("faces2_detected.jpg", img)