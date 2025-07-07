import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

# Pre-trained model paths
AGE_PROTO = "age_deploy.prototxt"
AGE_MODEL = "age_net.caffemodel"
GENDER_PROTO = "gender_deploy.prototxt"
GENDER_MODEL = "gender_net.caffemodel"

# Label definitions
AGE_LIST = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
GENDER_LIST = ['Male', 'Female']

# Load models
age_net = cv2.dnn.readNetFromCaffe(AGE_PROTO, AGE_MODEL)
gender_net = cv2.dnn.readNetFromCaffe(GENDER_PROTO, GENDER_MODEL)

# Load Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def detect_age_gender(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None  # Handle failed image read

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_img = image[y:y+h, x:x+w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227),
                                     (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Gender prediction
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = GENDER_LIST[gender_preds[0].argmax()]

        # Age prediction
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = AGE_LIST[age_preds[0].argmax()]

        label = f"{gender}, {age}"
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 200), 2)

    return image

def upload_and_process():
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
    if file_path:
        result_img = detect_age_gender(file_path)
        if result_img is not None:
            # Convert OpenCV BGR image to RGB and then to PIL
            img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(image=img_pil)

            image_label.config(image=img_tk)
            image_label.image = img_tk  # Prevent garbage collection
            status_label.config(text="Detection complete.")
        else:
            status_label.config(text="Failed to load image.")

# Tkinter setup
root = tk.Tk()
root.title("Age and Gender Detection")
root.geometry("800x600")

upload_button = tk.Button(root, text="Upload Image", command=upload_and_process)
upload_button.pack(pady=10)

status_label = tk.Label(root, text="No image uploaded")
status_label.pack()

image_label = tk.Label(root)
image_label.pack(pady=10)

root.mainloop()

