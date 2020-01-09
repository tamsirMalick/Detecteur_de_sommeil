"""Ce script utilise la haarcascade d'OpenCV (cascade du visage et des yeux) pour détecter le visage
et les yeux dans une image d'entrée donnée."""

# Import des librairies necessaires
import cv2 as cv


# Charger la cascade de visages et la cascade de cheveux à partir du dossier haarcascades
face_cascade = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier("haarcascades/haarcascade_eye.xml")

# On stock l'image dans un variable img, ensuite nous le convertissons en niveaux de gris pour le stocker dans une variable grey.
# L'image est convertie en niveaux de gris, car la cascade de visages ne nécessite pas de fonctionner sur des images colorées
img = cv.imread('images/test.jpeg')
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Détectez tous les visages dans l'image.
faces = face_cascade.detectMultiScale(gray, 1.3, 5)

# Nous dessinons un rectangle sur le visage pour faciliter la détection des yeux sur les visages
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # ROI est une région d'intérêt avec une zone à l'intérieur.
    roi_gray = gray[y:y + h, x:x + w]
    roi_color = img[y:y + h, x:x + w]

    # Détecter les yeux du visage
    eyes = eye_cascade.detectMultiScale(roi_gray)

    for (ex, ey, ew, eh) in eyes:
        cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

cv.imshow('Image', img)
cv.waitKey(0)
cv.destroyAllWindows()
