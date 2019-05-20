import time
import os
import cv2
# import picam

# Image configuration
image_extension = ".png"

def add_new_user(label):
    global image_extension

    path = os.getcwd() + "/images/" + label

    if not os.path.exists(path):
        os.makedirs(path)
        print("Created a new images folder called: " + label)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(base_dir, "images")
    image_dir = os.path.join(images_dir, label)

    print("In 5 seconds, the device will take a few pictures of you.")
    print("Please move your head in multiple angles for better results.")

    for second in range(5):
        print(5 - second)
        time.sleep(1)

    # camera = OpenCVCapture.read(0)
    camera = cv2.VideoCapture(0)
    for image_number in range(100):
        ret, frame = camera.read()
        image_name = label + str(image_number) + image_extension

        cv2.imwrite(os.path.join(image_dir, image_name), frame)
        print("Created image: " + image_name)
        time.sleep(0.5)
