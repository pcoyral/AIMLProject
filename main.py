# main.py (Streamlit App)
import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO

def process_video(video_path):
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        annotated_frame = results[0].plot()
        yield annotated_frame

st.title("Tennis Detection")
uploaded_file = st.file_uploader("Upload a tennis video", type=["mp4"])
if uploaded_file:
    with open("input.mp4", "wb") as f:
        f.write(uploaded_file.read())
    st.video("input.mp4")
    if st.button("Process"):
        st.write("Processing...")
        processed_frames = process_video("input.mp4")
        for frame in processed_frames:
            st.image(frame, channels="BGR")