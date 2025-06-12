# ExpressAI: Real-time and Video-based Face Emotion Recognition
ExpressAI is a web application that leverages a hybrid deep learning model to detect and classify human emotions from video input. It offers both real-time emotion detection using a webcam feed and analysis of uploaded video files.

**Features**
- Video File Analysis: Upload a video and get a summary of the dominant emotions detected, along with frames showcasing those emotions at their highest probability.
- Real-time Emotion Detection: Utilize your webcam for live emotion analysis, displaying the detected emotions and their probabilities.
Intuitive User Interface: A clean and responsive web interface built with Flask and HTML/CSS/JavaScript for easy interaction.
- Emotion Labels: Detects a range of emotions including: 'Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral'.
Progress Tracking: For video analysis, a progress bar and loading indicator keep you informed about the processing status.

**Technologies Used**
- Flask: Python web framework for the backend.
OpenCV (cv2): For video processing, frame extraction, and image manipulation.
- TensorFlow/Keras: For loading and running the pre-trained hybrid emotion recognition model.
- RetinaFace: (Assumed from RetinaFace.detect_faces) A powerful face detection library used for accurate face localization within frames.
HTML, CSS, JavaScript: For the frontend user interface.
- PIL (Pillow): For image processing, specifically converting OpenCV images to PIL format for model input.





