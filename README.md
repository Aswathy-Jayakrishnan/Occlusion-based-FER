ExpressAI: Real-time and Video-based Face Emotion Recognition
ExpressAI is a web application that leverages a hybrid deep learning model to detect and classify human emotions from video input. It offers both real-time emotion detection using a webcam feed and analysis of uploaded video files.

Features
Video File Analysis: Upload a video and get a summary of the dominant emotions detected, along with frames showcasing those emotions at their highest probability.
Real-time Emotion Detection: Utilize your webcam for live emotion analysis, displaying the detected emotions and their probabilities.
Intuitive User Interface: A clean and responsive web interface built with Flask and HTML/CSS/JavaScript for easy interaction.
Emotion Labels: Detects a range of emotions including: 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'.
Progress Tracking: For video analysis, a progress bar and loading indicator keep you informed about the processing status.
Technologies Used
Flask: Python web framework for the backend.
OpenCV (cv2): For video processing, frame extraction, and image manipulation.
TensorFlow/Keras: For loading and running the pre-trained hybrid emotion recognition model.
RetinaFace: (Assumed from RetinaFace.detect_faces) A powerful face detection library used for accurate face localization within frames.
HTML, CSS, JavaScript: For the frontend user interface.
PIL (Pillow): For image processing, specifically converting OpenCV images to PIL format for model input.

Setup and Installation

Follow these steps to get ExpressAI up and running on your local machine.

Prerequisites
Python 3.8+
pip (Python package installer)
1. Clone the Repository

git clone <your-repository-url>
cd ExpressAI
2. Create a Virtual Environment (Recommended)
Bash

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install Dependencies
Bash

pip install -r requirements.txt
(Note: You'll need to create a requirements.txt file. See the next section.)

4. Create requirements.txt
Based on your app.py, create a file named requirements.txt in the root directory of your project with the following content:

Flask
opencv-python
tensorflow
Pillow
retina-face  # Assuming this is the library you're using for RetinaFace
numpy
Note: The actual package name for RetinaFace might vary. Please verify the correct pip installable name for the RetinaFace library you are using. If it's part of deepface or a similar library, list that instead.

5. Place Your Model File
Ensure your trained hybrid emotion recognition model (hybrid_fer_model.h5) is placed in the root directory of your project, next to app.py.

6. Create Static and Uploads Folders
Ensure the following directories exist in your project's root:

.
├── app.py
├── hybrid_fer_model.h5
├── requirements.txt
├── static/
│   └── disp2.jpg  # Place your background image here
├── templates/
│   ├── index.html
│   ├── realtime.html
│   └── results.html
├── uploads/
├── results/
└── realtime_emotions/

7. Run the Application
Bash

python app.py
The application will typically run on http://127.0.0.1:5000/. Open this URL in your web browser.

Usage
Home Page (/)
The home page provides two main options:

  Upload Video: Click this button to reveal a form where you can upload a video file for emotion analysis. Supported formats include MP4, AVI, MOV, and MKV.

  Real-time Detection: Click this button to start real-time emotion detection using your webcam.

Video Analysis
Click on "Upload Video" on the home page.
Select a video file from your computer.
Click "Proceed".
The application will process the video, and a progress bar will be displayed.
Once processing is complete, you'll be redirected to the results page showing the detected emotions with corresponding frames and probabilities.

Real-time Detection
Click on "Real-time Detection" on the home page.
Your browser might ask for permission to access your webcam. Grant it.
The live webcam feed will appear, and detected emotions will be displayed alongside their probabilities.

Folder Structure

app.py: The main Flask application.

hybrid_fer_model.h5: Your pre-trained emotion recognition model.

static/: Contains static files like CSS, JavaScript, and images (e.g., disp2.jpg for backgrounds).

templates/: Contains HTML template files:

index.html: The main landing page.

realtime.html: Page for real-time webcam detection.

results.html: Page displaying video analysis results.

uploads/: Temporary directory to store uploaded video files.

results/: (Currently unused in the provided code, but good practice to keep) Could be used to save analysis reports or processed frames.

realtime_emotions/: Stores text files with timestamps, logging real-time detected emotions and their probabilities.
