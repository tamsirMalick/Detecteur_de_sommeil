"""Ce script détecte si une personne est somnolente ou non, en utilisant dlib et le rapport hauteur / largeur des yeux.
On utilise le flux vidéo de la webcam comme entrée."""

# Import des librairies necessaires
from scipy.spatial import distance
from imutils import face_utils
import pygame
import time
import dlib
import cv2

# Initialiser pygame et charger le son de l'alarme
pygame.mixer.init()
pygame.mixer.music.load('audio/alert.wav')

# Seuil minimal du rapport hauteur / largeur sous lequel l'alarme est déclenchée
EYE_ASPECT_RATIO_THRESHOLD = 0.3

# Trames consécutives minimales pour lesquelles le rapport oculaire est inférieur au seuil de déclenchement de l'alarme
EYE_ASPECT_RATIO_CONSEC_FRAMES = 50

# compteur de trames consécutives sous la valeur seuil
COUNTER = 0

# Charger la cascade de faces qui sera utilisée pour dessiner un rectangle autour des faces détectées.
face_cascade = cv2.CascadeClassifier("haarcascades/haarcascade_frontalface_default.xml")


# Cette fonction calcule et renvoie le rapport hauteur / largeur des yeux
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])

    ear = (A + B) / (2 * C)
    return ear


# Détecteur et prédicteur de face de charge, utilise un fichier de prédicteur de forme dlib
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Extraire les index des repères faciaux pour l'œil gauche et droit
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# Démarrer la capture vidéo webcam
video_capture = cv2.VideoCapture(0)

# Donner du temps à la caméra pour s'initialiser (non requis)
time.sleep(2)

while True:
    # On Lit chaque image, on la convertie puis on la retourne en niveaux de gris
    ret, frame = video_capture.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Détecter les points faciaux grâce à la fonction de détection
    faces = detector(gray, 0)

    # Détecter les visages via haarcascade_frontalface_default.xml
    face_rectangle = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Nous Dessinons un rectangle autour de chaque visage détecté
    for (x, y, w, h) in face_rectangle:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Détecter les points faciaux
    for face in faces:
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # On recupere le tableau contenant les coordonnées de l'œil gauche et de l'œil droit
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        # Calculer le rapport d'aspect des deux yeux
        leftEyeAspectRatio = eye_aspect_ratio(leftEye)
        rightEyeAspectRatio = eye_aspect_ratio(rightEye)

        eyeAspectRatio = (leftEyeAspectRatio + rightEyeAspectRatio) / 2

        # On Utilisez la fonction "convexHull" pour éliminer les écarts de contour convexes et dessiner la forme des yeux autour des yeux
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # On cherche si le rapport hauteur / largeur des yeux est inférieur au seuil
        if eyeAspectRatio < EYE_ASPECT_RATIO_THRESHOLD:
            COUNTER += 1
            # Si non, le trames est supérieur aux trames de seuil,
            if COUNTER >= EYE_ASPECT_RATIO_CONSEC_FRAMES:
                pygame.mixer.music.play(-1)
                cv2.putText(frame, "Vous êtes en train de dormir", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 2)
        else:
            pygame.mixer.music.stop()
            COUNTER = 0

    # Afficher le flux vidéo
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Enfin, une fois la capture vidéo terminée, On ferme la fenetre de capture
video_capture.release()
cv2.destroyAllWindows()
