import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile

# Constants
DEMO_IMAGE = 'demo.jpg'  # Make sure to have a demo image in your directory

BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

# Initialize neural network
def load_network():
    net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
    return net

# Pose detection function
def detect_pose(frame, net, threshold=0.2):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (368, 368), 
                                      (127.5, 127.5, 127.5), swapRB=True, crop=False))
    
    out = net.forward()
    out = out[:, :19, :, :]
    
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > threshold else None)
    
    # Draw the detected points and lines
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        
        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
    
    return frame

def main():
    st.title("Human Pose Estimation Application")
    st.write("Upload an image or video to detect human poses")
    
    # Load the neural network
    net = load_network()
    
    # File type selection
    file_type = st.radio("Select input type:", ("Image", "Video"))
    
    if file_type == "Image":
        # Image upload
        img_file_buffer = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        
        if img_file_buffer is not None:
            # Convert uploaded file to image
            image = np.array(Image.open(img_file_buffer))
        else:
            # Use demo image
            try:
                image = np.array(Image.open(DEMO_IMAGE))
            except:
                st.error("Please upload an image or ensure demo image exists in directory")
                return
        
        # Create a slider for detection threshold
        threshold = st.slider('Detection Threshold', 0.0, 1.0, 0.2, 0.05)
        
        # Show original image
        st.subheader("Original Image")
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Process image and show result
        if st.button('Detect Poses'):
            result_image = detect_pose(image.copy(), net, threshold)
            st.subheader("Detection Result")
            st.image(result_image, caption="Detected Poses", use_column_width=True)
    
    else:  # Video processing
        video_file_buffer = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        
        if video_file_buffer is not None:
            # Save uploaded video to temporary file
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file_buffer.read())
            
            # Process video
            cap = cv2.VideoCapture(tfile.name)
            
            # Video information
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create video writer
            temp_output_path = "output_video.mp4"
            out = cv2.VideoWriter(temp_output_path, 
                                cv2.VideoWriter_fourcc(*'mp4v'),
                                fps, (frame_width, frame_height))
            
            # Progress bar
            progress_bar = st.progress(0)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Process each frame
            for i in range(frame_count):
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                processed_frame = detect_pose(frame, net)
                out.write(processed_frame)
                
                # Update progress bar
                progress_bar.progress((i + 1) / frame_count)
            
            cap.release()
            out.release()
            
            # Display processed video
            st.video(temp_output_path)
            
            # Cleanup
            import os
            os.unlink(tfile.name)
            os.unlink(temp_output_path)

if __name__ == '__main__':
    main()