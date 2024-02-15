# MULTISENSORY APPROACH TO ENHANCE DIGITAL LITERATURE CONSUMPTION



## Introduction:


In an era marked by the ever-evolving landscape of digital literature, the "Multisensory Approach to Enhance Digital Literature Consumption" stands as a groundbreaking initiative poised to redefine the very essence of reading. This innovative project represents a seamless fusion of technology, literature, and music, introducing a transformative experience that transcends the conventional boundaries of textual engagement. At its core, the project leverages the power of multisensory stimuli, tracking readers' eye movements, and employing sentiment analysis to create a personalized and emotionally resonant reading journey.

The project's inception involves a meticulous calculation of the average time readers take to traverse a line, a metric that serves as the cornerstone for delivering a tailor-made reading experience. This fundamental data is then employed in a groundbreaking marriage of literature and music, as the chosen e-version of the text undergoes sentiment analysis on a paragraph-by-paragraph basis. This analysis unveils the emotional nuances intricately woven into the narrative, setting the stage for a harmonious symphony that complements the reader's emotional and pacing preferences.

The orchestration of this musical accompaniment is no arbitrary feat; it is meticulously curated from a database designed to resonate with the sentiment and pacing of each paragraph. Crucially, the music synchronizes with the reader's average reading speed for each line, culminating in an immersive experience where literature, technology, and music coalesce to craft an artful and emotionally rich reading journey.
Beyond its aesthetic appeal, the "Multisensory Approach to Enhance Digital Literature Consumption" offers a plethora of applications. In educational settings, it proves invaluable in enhancing comprehension of complex materials and fostering language learning. Simultaneously, its impact extends to professional contexts, refining focus during extensive reading tasks. As this project paves the way for the future of digital literature consumption, it paints a vivid picture of reading as not merely a cognitive exercise but as an art form—one that is immersive, emotionally charged, and deeply engaging.


## Features:
The proposed "Multisensory Approach to Enhance Digital Literature Consumption" system adopts a
systematic methodology that seamlessly integrates eye gaze direction analysis, text extraction through OCR, sentimental analysis using RoBERTa, dynamic music selection, and personalized music playback duration calculation.

### Eye Gaze Direction Analysis:
The system initiates by employing precise face and eye detection through the dlib library,
emphasizing the identification of crucial facial features, particularly focusing on the eyes. This initial step lays the groundwork for understanding the user's eye movements and forms the basis for subsequent analyses.

### Gaze Ratio Calculation:
Following eye detection, a dedicated function is implemented to calculate the gaze ratio based
on identified eye landmarks. This calculation offers insights into the distribution of white pixels in both the left and right eyes. The meticulous analysis of gaze ratios is a critical process that provides valuable information for determining the direction of gaze—whether it is directed to the left or right.
   
### Reading Speed Calculation:
The system further refines its analysis by incorporating a timer mechanism. This timer is initiated when the gaze shifts to the right and stops when the gaze returns to the left, capturing the time taken for each iteration. This meticulous timing process serves as the foundation for understanding and calculating the user's reading speed. The precise measurement of the time spent on each iteration contributes to the system's ability to provide a personalized reading experience

### Text Extraction Using OCR
In the next phase, the system seamlessly integrates Tesseract OCR to extract textual content from images uploaded by the user. This step ensures an efficient transition from image-based literature to machine-readable text, setting the stage for subsequent sentiment analysis.


### Sentimental Analysis Using RoBERTa
The sentimental analysis pipeline is established using the transformers library, incorporating the RoBERTa-based model (arpanghoshal/EmoRoBERTa). The extracted text undergoes sentiment analysis, providing the system with emotion labels crucial for the dynamic selection of accompanying music.
    
## Requirements
### Hardware Requirements
    Computer or Laptop:
    Speakers or Headphones (Optional):
### Softare Requirements

#### Operating System:  
Windows, macOS, or Linux.
#### Development Environment:  
Visual Studio Code, PyCharm, or others.
#### Programming Language: 
Python, and relevant libraries and frameworks.
#### Image Processing Libraries: 
Integration of image processing libraries like OpenCV for tasks such as blurring, unblurring, and image manipulation.
#### Text Recognition Library: 
Integration of OCR (Optical Character Recognition) libraries such as Tesseract to accurately recognize and extract text from images.
### Deep Learning Frameworks:
Installation of deep learning frameworks, specifically dlib and transformers, is essential.
These frameworks leverage pre-trained models for facial landmark detection and emotion
analysis, enhancing the accuracy of the system.


## Program
```python
import cv2
import dlib
import numpy as np
import time

# Load the face detector and landmark predictor
face_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Initialize your video capture (replace '0' with the video file path or camera index)
cap = cv2.VideoCapture(0)

# Create a resizable window for the full face
cv2.namedWindow("Full Face", cv2.WINDOW_NORMAL)

# Define the font for displaying text
font = cv2.FONT_HERSHEY_SIMPLEX

def calculate_gaze_ratio(eye_points, landmarks):
    # Extract coordinates of the eye landmarks
    eye_coordinates = [(landmarks.part(i).x, landmarks.part(i).y) for i in eye_points]

    # Extract the eye region
    eye_x = min(x for x, y in eye_coordinates)
    eye_y = min(y for x, y in eye_coordinates)
    eye_width = max(x for x, y in eye_coordinates) - eye_x
    eye_height = max(y for x, y in eye_coordinates) - eye_y
    eye_region = gray[eye_y:eye_y + eye_height, eye_x:eye_x + eye_width]

    # Threshold the eye region with values 70, 255
    _, thresholded_eye = cv2.threshold(eye_region, 70, 255, cv2.THRESH_BINARY)

    # Create a mask with the same size as the eye region
    mask = np.zeros_like(eye_region)
    thresholded_eye = cv2.resize(thresholded_eye, None, fx=5, fy=5)
    height, width = thresholded_eye.shape
    left_side_threshold = thresholded_eye[0: height, 0:int(width/2)]
    left_side_white = cv2.countNonZero(left_side_threshold)
    right_side_threshold = thresholded_eye[0: height, int(width/2): width]
    right_side_white = cv2.countNonZero(right_side_threshold)

    # Check if right_side_white is zero to avoid division by zero
    if right_side_white == 0:
        return float('inf')  # Return positive infinity
    else:
        gaze_ratio = left_side_white / right_side_white
        return gaze_ratio

# Initialize variables for timer
start_time = None
end_time = None
total_time = 0
iteration_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale for face and eye detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_detector(gray)

    for face in faces:
        
        # Get facial landmarks (including eyes)
        landmarks = shape_predictor(gray, face)

        # Define the left eye points (points 36 to 41)
        left_eye_points = list(range(36, 42))

        # Define the right eye points (points 42 to 47)
        right_eye_points = list(range(42, 48))

        # Calculate the gaze ratio for the left and right eyes
        left_gaze_ratio = calculate_gaze_ratio(left_eye_points, landmarks)
        right_gaze_ratio = calculate_gaze_ratio(right_eye_points, landmarks)
        
        # Calculate the direction ratio
        direction_ratio = (left_gaze_ratio + right_gaze_ratio) / 2.0

        # Determine gaze direction based on direction ratio
        gaze = ""
        if 0.3 < direction_ratio < 0.55:
            gaze = "left"
        elif direction_ratio > 2.0:
            gaze = "right"

        # Start the timer when gaze goes to the right
        if gaze == "right" and start_time is None:
            start_time = time.time()

        # Stop the timer when gaze reaches the left
        if gaze == "left" and start_time is not None:
            end_time = time.time()
            iteration_time = end_time - start_time
            total_time += iteration_time
            iteration_count += 1
            start_time = None
        
        # Display the gaze ratios and gaze direction on the frame
        cv2.putText(frame, f"Left Gaze Ratio: {left_gaze_ratio:.2f}", (50, 100), font, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Right Gaze Ratio: {right_gaze_ratio:.2f}", (50, 150), font, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Direction Ratio: {direction_ratio:.2f}", (50, 200), font, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Gaze Direction: {gaze}", (50, 250), font, 1, (0, 0, 255), 2)

    # Display the full face in the main window
    cv2.imshow("Full Face", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Calculate and print the average time per iteration
if iteration_count > 0:
    average_time = total_time / iteration_count
    print(f"Average Time Per Iteration: {average_time:.2f} seconds")

cap.release()
cv2.destroyAllWindows()

import pytesseract
from transformers import pipeline
from IPython.display import Audio, display
import os
from pydub import AudioSegment
import time

# Load the image
image_path = 'largepreview.png'  # Replace with the path to your image
img = cv2.imread(image_path)

# Perform OCR to extract text from the image
text = pytesseract.image_to_string(img)

# Split the text into paragraphs based on consecutive non-empty lines
paragraphs = [paragraph.strip() for paragraph in text.split('\n\n') if paragraph.strip()]

# Initialize the emotion list to store detected emotions for each paragraph
emotion_list = []

# Define the emotion detection pipeline
emotion_pipeline = pipeline('sentiment-analysis', model='arpanghoshal/EmoRoBERTa')

# Map emotion labels to the corresponding emotion names in your dataset
emotion_mapping = {
    "admiration": "Admiration",
    "amusement": "Amusement",
    "anger": "Anger",
    "annoyance": "Annoyance",
    "approval": "Approval",
    "caring": "Caring",
    "confusion": "Confusion",
    "curiosity": "Curiosity",
    "desire": "Desire",
    "disappointment": "Disappointment",
    "disapproval": "Disapproval",
    "disgust": "Disgust",
    "embarrassment": "Embarrassment",
    "excitement": "Excitement",
    "fear": "Fear",
    "gratitude": "Gratitude",
    "grief": "Grief",
    "joy": "Joy",
    "love": "Love",
    "nervousness": "Nervousness",
    "optimism": "Optimism",
    "pride": "Pride",
    "realization": "Realization",
    "relief": "Relief",
    "remorse": "Remorse",
    "sadness": "Sadness",
    "surprise": "Surprise",
    "neutral": "Neutral"
}

# Iterate through each paragraph
for i, paragraph in enumerate(paragraphs, start=1):
    lines = paragraph.split('\n')
    num_lines = len(lines)

    # Perform emotion detection for the current paragraph
    emotion_labels = emotion_pipeline(paragraph)
    emotion_label = emotion_labels[0]['label']

    # Append the detected emotion to the emotion list
    emotion_list.append(emotion_label)

    # Get the corresponding emotion name
    emotion_name = emotion_mapping.get(emotion_label, "Unknown")

    # Play music for the corresponding emotion
    music_duration = 4.3 * num_lines
    file_path = os.path.join(music_directory, f"{emotion_name}.mp3")

    # Display the audio with a specified duration
    audio_data = open(file_path, 'rb').read()
    display(Audio(audio_data, rate=44100, autoplay=True), metadata={'duration': music_duration})

    # Introduce a delay to ensure the previous audio segment stops playing
    time.sleep(music_duration)

# Concatenate music based on the emotion list using pydub
concatenated_music = AudioSegment.silent()
for emotion_label in emotion_list:
    emotion_name = emotion_mapping.get(emotion_label, "Unknown")
    file_path = os.path.join(music_directory, f"{emotion_name}.mp3")
    audio_clip = AudioSegment.from_file(file_path, format="mp3")
    concatenated_music += audio_clip

# Convert the combined audio to binary data
combined_audio_data = concatenated_music.raw_data
combined_audio_rate = concatenated_music.frame_rate

# Display the combined audio with a specified duration
combined_music_duration = concatenated_music.duration_seconds
display(Audio(combined_audio_data, rate=combined_audio_rate, autoplay=True), metadata={'duration': combined_music_duration})

```


## Output

### Average reading speed of the user
![image](https://github.com/P-S-Pradeep/Multi-Sensory-Literature/assets/102652887/815f4045-2231-44b3-bd70-44ffa2c9a722)


### Text extraction after OCR
![image](https://github.com/P-S-Pradeep/Multi-Sensory-Literature/assets/102652887/18f35e8c-0fcc-49f0-b3aa-ab49e864bcb7)

### Combined audio based on emotion
![image](https://github.com/P-S-Pradeep/Multi-Sensory-Literature/assets/102652887/dd7dea97-b7b9-44eb-90df-21884304c6d9)

## Result


This project introduces a novel approach to enhance the digital reading experience through dynamic bounding box text unblurring. The methodology developed utilizes OCR technology and bounding boxes to isolate individual lines of text within e-books and digital content. By dynamically unblurring text based on the reader’s speed, this approach addresses the common issues of reader distraction and the loss of one's place within a text.
The testing and evaluation of the system demonstrate its potential to significantly improve reading comprehension and reduce distractions, ultimately providing a more user-friendly and immersive digital reading experience. The interactive features allow readers to tailor their reading experience to their preferences.As e-books and digital content continue to gain popularity, this project offers a practical and effective solution to enhance the quality of digital reading..

