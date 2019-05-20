import RPi.GPIO as GPIO
import cv2
import pickle
# import picam

# Setup Raspberry LEDs
green_led_pin = 18
red_led_pin = 16

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
GPIO.setup(green_led_pin, GPIO.OUT)
GPIO.setup(red_led_pin, GPIO.OUT)

face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained-faces.yml")

# Name text style configuration
name_font = cv2.FONT_HERSHEY_SIMPLEX
name_color = (255, 255, 255)  # white
name_stroke = 1
name_fontsize = 0.7

# Detected area style configuration
area_color = (25, 25, 255)  # light red
area_stroke = 2

# Get every label and it's id
labels = {}
with open("labels.pickle", 'rb') as file:
    unordered_labels = pickle.load(file)
    labels = {
        v:k for k, v in unordered_labels.items()
    }

# camera = OpenCVCapture.read(0)
camera = cv2.VideoCapture(0)

while True:
    ret, frame_color = camera.read()

    # Convert frame to grayscale for better detection
    frame_grayscale = cv2.cvtColor(frame_color, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(frame_grayscale, scaleFactor=1.5, minNeighbors=5)

    for (x, y, w, h) in faces:
        region_of_interest = frame_grayscale[y:y+h, x:x+w]

        # Recognize faces using deep learning model
        predicted_user_id, confidence = recognizer.predict(region_of_interest)
        if confidence <= 75:
            GPIO.output(green_led_pin, GPIO.HIGH)
            GPIO.output(red_led_pin, GPIO.LOW)
            name = labels[predicted_user_id]

            # If a user is detected, display the name above the detected area
            cv2.putText(frame_color, name, (x, y - 10), name_font, name_fontsize, name_color, name_stroke, cv2.LINE_AA)
        else:
            GPIO.output(green_led_pin, GPIO.LOW)
            GPIO.output(red_led_pin, GPIO.HIGH)

        # Draw a rectangle around the detected face
        cv2.rectangle(frame_color, (x, y), (x + w, y + h), area_color, area_stroke)

    # Print as live camera
    cv2.imshow('frame', frame_color)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
