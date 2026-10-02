# Indian Sign Language Translator

## 📌 Project Overview

The **Indian Sign Language Translator** is a Python-based computer vision and machine learning project designed to recognize Indian Sign Language gestures from a live webcam feed and convert the detected signs into understandable text.

The project uses **OpenCV** for processing the webcam input and machine-learning-based image recognition techniques for identifying hand gestures. The recognized output can also be converted into **speech**, making the system useful as a communication-assistance application.

## 🎯 Objectives

The main objectives of this project are:

- Detect hand gestures using a webcam.
- Recognize predefined Indian Sign Language signs.
- Convert recognized gestures into text.
- Provide speech output for the recognized text.
- Process gestures from a live camera feed.
- Create a simple and accessible communication system.

## ✨ Features

- 🎥 **Live Webcam Detection**
  - Captures hand gestures through the computer's webcam.

- ✋ **Hand Gesture Recognition**
  - Detects and processes hand gestures from the camera feed.

- 🔤 **Sign-to-Text Conversion**
  - Converts recognized signs into corresponding text.

- 🔊 **Text-to-Speech**
  - Converts the recognized text into audio output.

- ⚡ **Near Real-Time Processing**
  - Processes webcam frames continuously to provide immediate recognition.

- 🖥️ **Desktop-Based Application**
  - Runs locally using Python and a webcam.

## 🛠️ Technologies Used

### Programming Language
- Python

### Libraries and Technologies
- OpenCV
- TensorFlow
- Keras
- NumPy
- Pillow

### Machine Learning
- Image classification
- Pre-trained/trained deep-learning model
- Computer vision-based gesture recognition

## 🔄 How the System Works

The basic workflow of the application is:

```text
Webcam
   ↓
Live Video Frame
   ↓
Image/Hand Processing
   ↓
Preprocessing
   ↓
Machine Learning Model
   ↓
Sign Recognition
   ↓
Recognized Text
   ↓
Text-to-Speech
   ↓
Audio Output
```

### Step 1 — Webcam Input

The application accesses the computer webcam and continuously captures video frames.

### Step 2 — Image Processing

The captured frames are processed using OpenCV to prepare the image for recognition.

### Step 3 — Hand Gesture Detection

The system identifies the relevant hand region from the camera input and prepares it for classification.

### Step 4 — Sign Recognition

The processed image is provided to the trained machine-learning model. The model predicts the corresponding sign from the supported classes.

### Step 5 — Text Output

The predicted sign is displayed as text on the application interface.

### Step 6 — Speech Output

The recognized text can be converted into speech so that the detected sign can be heard as audio.

## 📂 Project Structure

```text
Indian-Sign-Language-Translator/
│
├── App/
│   ├── main.py
│   ├── final_model
│   └── ...
│
├── Dataset/
│   └── ...
│
├── Model/
│   └── ...
│
├── requirements.txt
│
└── README.md
```

> The exact folders and files may vary depending on the final version of the project uploaded to GitHub.

## ⚙️ Requirements

Before running the project, install Python and the required dependencies.

Install the required packages using:

```bash
pip install -r requirements.txt
```

Or install the main libraries individually:

```bash
pip install opencv-python
pip install tensorflow
pip install keras
pip install numpy
pip install pillow
```

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/jiv143/Indian-Sign-Language-Translator.git
```

### 2. Open the project directory

```bash
cd Indian-Sign-Language-Translator
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

Navigate to the application directory and run:

```bash
python main.py
```

### 5. Allow Webcam Access

Make sure your computer has a working webcam and allow the application to access it.

The application will capture the hand gesture and display the predicted sign.

## 📊 Project Output

The application is designed to provide:

```text
Hand Gesture
     ↓
Recognized Sign
     ↓
Text Output
     ↓
Speech Output
```

For example:

```text
Input:  Hand Gesture
Output: Recognized Sign
        ↓
        Text
        ↓
        Voice
```

The exact signs recognized depend on the classes supported by the trained model used in the project.

## 💡 Applications

The project can be useful in areas such as:

- Communication assistance
- Educational applications
- Accessibility-focused software
- Human-computer interaction
- Sign language learning
- Assistive technology
- Computer vision research

## 🚀 Future Improvements

The project can be further improved by:

- Increasing the number of recognized signs.
- Improving recognition accuracy.
- Supporting continuous sign-language sentences.
- Improving performance under different lighting conditions.
- Improving hand detection for different backgrounds.
- Developing a web-based version.
- Developing an Android/mobile application.
- Adding multilingual speech output.
- Improving the user interface.
- Supporting more complex gestures.

## 📚 Learning Outcomes

Through this project, I gained practical exposure to:

- Python programming
- Computer vision
- OpenCV
- Image preprocessing
- Machine learning
- Deep learning concepts
- Model integration
- Webcam-based applications
- Text-to-speech systems
- Building and running a machine-learning application

## 👨‍💻 Author

**Jivan Desai**

MCA Student | Computer Science

GitHub:  
https://github.com/jiv143

LinkedIn:  
https://linkedin.com/in/jivan-desai-9538a5332

## 📄 Disclaimer

This project is an academic and learning-oriented implementation of an Indian Sign Language recognition system. Recognition performance depends on the trained model, supported gesture classes, camera quality, lighting conditions, and input environment.