import streamlit as st
from PIL import Image
import io
import pandas as pd
import cv2
import tensorflow as tf
import numpy as np
import sys
from time import sleep

sys.stdout.reconfigure(encoding="utf-8")
st.set_page_config(page_title="Face recognition app")
class_names = ["angry", "disgusted", "scared", "happy", "neutral", "sad", "surprised"]
model = tf.keras.models.load_model("face_recognition.keras")


def evaluate_photo(photo):
    image = Image.open(photo)
    image_np = np.array(image)
    input_image = image_np.reshape((1, 48, 48, 1)).astype("float32") / 255.0
    result = model.predict(input_image)
    pred_labels = np.argmax(result, axis=1)

    return class_names[int(pred_labels[0])]


def write_text_in_center(text, header):
    st.markdown(
        f"<{header} style='text-align: center;'>{text}</{header}>",
        unsafe_allow_html=True,
    )


write_text_in_center("Hello!", "h1")
write_text_in_center(
    "This is a simple program for evaluating face emotions based on photo!", "h2"
)
write_text_in_center("Please import your file below.", "h3")


photo_uploader = st.file_uploader("Upload photo", type=["jpg", "jpeg", "png"])
write_text_in_center("Or", "h5")
camera_button = st.button("Take a selfie!", use_container_width=True)

if camera_button:
    camera_photo = st.camera_input("Take a picture")
    st.write(camera_photo is None)
    if camera_photo is not None:
        write_text_in_center(evaluate_photo(camera_photo), "h4")
        # sleep(5)

if photo_uploader is not None:
    st.image(photo_uploader, use_column_width="always")
    write_text_in_center(evaluate_photo(photo_uploader), "h4")

evaluate_photo = st.button("Evaluate!", use_container_width=True)
