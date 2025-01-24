
# Human Pose Estimation Using Machine Learning

This project demonstrates human pose estimation using machine learning techniques with OpenCV and Streamlit. The application provides two primary functionalities:
1. Estimating poses in video files.
2. Detecting and estimating poses in still images.

## Features
- Processes video files to estimate human poses frame by frame.
- Detects key points and draws skeletons on uploaded images.
- Interactive Streamlit interface for easy use.
- Adjustable threshold settings for pose detection.

## Requirements
The following Python dependencies are required to run the project:

```
opencv-python-headless==4.5.1.48
streamlit==0.76.0
numpy==1.18.5
matplotlib==3.3.2
Pillow==8.1.2
```

To install these dependencies, use the command:

```bash
pip install -r requirements.txt
```

## Installation
1. Clone this repository or download the project files.
2. Ensure you have Python 3.7 or higher installed on your system.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Place the pre-trained TensorFlow model file (`graph_opt.pb`) in the same directory as the Python scripts.

## Running the Applications

### 1. Pose Estimation on Videos
To run the application for video-based pose estimation, execute the following command:

```bash
streamlit run pose_estimation_Video.py
```

**Steps to use:**
1. Upload a video file in `.mp4`, `.avi`, or `.mov` format using the file uploader.
2. Click the "Process Video" button to start analyzing poses in the video.
3. The application will display the video frames with estimated poses in real time.

### 2. Pose Estimation on Images
To run the application for image-based pose estimation, execute the following command:

```bash
streamlit run estimation_app.py
```

**Steps to use:**
1. Upload an image in `.jpg`, `.jpeg`, or `.png` format using the file uploader.
2. Adjust the detection threshold using the provided slider.
3. The application will display the original image alongside the image with estimated poses.

## Directory Structure
```
.
├── pose_estimation_Video.py   # Script for video-based pose estimation
├── estimation_app.py          # Script for image-based pose estimation
├── requirements.txt           # Dependencies for the project
├── graph_opt.pb               # Pre-trained TensorFlow graph for pose estimation (to be added manually)
```

## Known Issues
- **Visibility:** Ensure the input images and videos have good visibility of all body parts for accurate detection.
- **Model File:** The project requires the `graph_opt.pb` file, which is not included in the repository. You need to download it separately and place it in the same directory as the Python scripts.

## Contributions
Contributions to the project are welcome. Feel free to create pull requests or report issues to improve the project.

## License
This project is licensed under the MIT License.

