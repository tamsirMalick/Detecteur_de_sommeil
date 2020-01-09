"""Ce script utilise la haarcascade d'OpenCV (cascade du visage et des yeux) pour détecter le visage
et les yeux dans une image d'entrée donnée."""

# Import des librairies necessaires
import cv2 as cv

# Charger la cascade de visages et la cascade de cheveux à partir du dossier haarcascades
face_cascade = cv.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")
eye_cascade = cv.CascadeClassifier("haarcascades/haarcascade_eye.xml")

# Capture d'une vidéo à partir d'une webcam
video_capture = cv.VideoCapture(0)

# Lire tous les cadres de la webcam
while True:
    ret, frame = video_capture.read()
    frame = cv.flip(frame, 1)  # On retourne le flux vidéo de sorte qu'il ne soit pas retourné et ressemble à un miroir.
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        for (ex, ey, ew, eh) in eyes:
            cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv.imshow('Video', frame)

    # Si l'utilisateur apuis sur la touche q, on quite le programme
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Enfin, une fois la capture vidéo terminée, On ferme la fenetre de capture
video_capture.release()
cv.destroyAllWindows()
