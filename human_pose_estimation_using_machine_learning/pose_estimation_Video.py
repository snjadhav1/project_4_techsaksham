import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from contextlib import contextmanager

class PoseEstimationApp:
    def __init__(self):
        self.BODY_PARTS = {
            "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
            "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
            "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
            "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
        }

        self.POSE_PAIRS = [
            ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
            ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
            ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
            ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
            ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
        ]
    
        self.width = 368
        self.height = 368
        self.inWidth = self.width
        self.inHeight = self.height
        self.thres = 0.2
        
        # Load neural network
        try:
            self.net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
            st.success("Neural network loaded successfully")
        except Exception as e:
            st.error(f"Error loading neural network: {e}")

    def process_frame(self, frame):
        """Process a single frame with pose estimation"""
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        # Prepare input blob
        net_input = cv2.dnn.blobFromImage(frame, 2.0, (self.inWidth, self.inHeight),
                                        (127.5, 127.5, 127.5), swapRB=True, crop=False)
        self.net.setInput(net_input)
        out = self.net.forward()
        out = out[:, :19, :, :]

        # Find points
        points = []
        for i in range(len(self.BODY_PARTS)):
            heatMap = out[0, i, :, :]
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            points.append((int(x), int(y)) if conf > self.thres else None)

        # Draw skeleton
        for pair in self.POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            idFrom = self.BODY_PARTS[partFrom]
            idTo = self.BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

        return frame

    @contextmanager
    def safe_temp_file(self, uploaded_file):
        """Safely handle temporary file creation and cleanup"""
        temp_dir = tempfile.mkdtemp()
        temp_path = os.path.join(temp_dir, "temp_video.mp4")
        
        try:
            with open(temp_path, 'wb') as f:
                f.write(uploaded_file.read())
            yield temp_path
        finally:
            try:
                # Clean up resources
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
            except Exception as e:
                st.warning(f"Note: Some temporary files may remain: {str(e)}")

    def run(self):
        """Run the Streamlit application"""
        st.title("Pose Estimation App")
        
        # File uploader
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
        
        if uploaded_file is not None:
            # Process video button
            if st.button('Process Video'):
                with self.safe_temp_file(uploaded_file) as temp_video_path:
                    try:
                        # Video capture
                        cap = cv2.VideoCapture(temp_video_path)
                        
                        if not cap.isOpened():
                            st.error("Failed to open video file")
                            return

                        # Create a placeholder for the video frames
                        stframe = st.empty()
                        
                        # Progress bar
                        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        progress_bar = st.progress(0)
                        
                        frame_count = 0
                        
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                                
                            # Process the frame
                            processed_frame = self.process_frame(frame)
                            
                            # Convert BGR to RGB for displaying in Streamlit
                            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                            
                            # Display the frame
                            stframe.image(processed_frame_rgb)
                            
                            # Update progress bar
                            frame_count += 1
                            progress_bar.progress(min(1.0, frame_count / total_frames))
                        
                        cap.release()
                        st.success("Video processing completed!")
                        
                    except Exception as e:
                        st.error(f"Error processing video: {str(e)}")
                        if 'cap' in locals():
                            cap.release()

if __name__ == "__main__":
    app = PoseEstimationApp()
    app.run()