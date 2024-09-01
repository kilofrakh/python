

# Face Recognition and Smile Detection Web Application

![License](https://img.shields.io/badge/license-MIT-green)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)
![Flask Version](https://img.shields.io/badge/flask-2.0.3-blue)
![OpenCV Version](https://img.shields.io/badge/opencv-4.5.5-orange)

A real-time face recognition and smile detection web application built using Flask, OpenCV, and scikit-learn. This project showcases how machine learning and computer vision techniques can be used to create interactive and engaging applications with a simple and intuitive web interface.

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Demo](#demo)
- [Installation](#installation)
- [Usage](#usage)
- [Technologies Used](#technologies-used)
- [How It Works](#how-it-works)
- [File Structure](#file-structure)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Overview

The Face Recognition and Smile Detection Web Application is a Python-based project that uses machine learning techniques to identify and recognize faces in real time. The system uses a webcam to capture images, detect faces, recognize identities, and detect smiles. It is designed to be easy to set up and extend, making it a great starting point for developers interested in computer vision and machine learning applications.

This application is primarily intended for educational purposes and demonstrates a practical use case of K-Nearest Neighbors (KNN) for face recognition, combined with OpenCV’s Haar cascades for face and smile detection. It can be further enhanced to include more sophisticated recognition techniques or be integrated into larger systems.

## Features

- **Face Data Collection**: Allows users to capture and save facial data directly from their webcam.
- **Real-Time Face Recognition**: Identifies known faces using a trained KNN model.
- **Smile Detection**: Detects smiles in real time and provides feedback on the screen.
- **Web-Based Interface**: A user-friendly interface built with Flask, making it easy to navigate and interact with.
- **Scalable and Modular**: Designed with extensibility in mind, allowing easy integration of additional features such as eye detection, age estimation, or more advanced recognition models.
- **Cross-Platform Compatibility**: Runs on Windows, macOS, and Linux with minimal setup.

## System Architecture

The application follows a modular architecture, ensuring that each component is separated and can be modified independently. The primary components include:

1. **Web Interface**: Built using Flask, it provides the front-end for users to interact with the system.
2. **Face Detection and Recognition**: Uses OpenCV to detect faces and scikit-learn's KNN for recognizing faces.
3. **Smile Detection**: Uses Haar cascades to identify smiles within detected faces.
4. **Data Management**: Facial data and names are stored using Python’s `pickle` module for easy loading and updating.

### Architecture Diagram

```plaintext
+-------------------+         +------------------------+           +------------------+
|   Web Interface   | <---->  |  Face Recognition/     |  <---->   |  Smile Detection |
| (Flask, HTML/CSS) |         |   Data Collection      |           |  (OpenCV Haar)   |
+-------------------+         +------------------------+           +------------------+
                                |                                     
                                v                                     
                       +-----------------------+
                       |   Data Management     |
                       | (Pickle Serialization)|
                       +-----------------------+
```

## Demo

To see the application in action, you can view a demo video [here](https://youtu.be/BB6jkStwweo?si=n5_dSMJsPmq6KP-L) . The demo showcases how the system detects faces, recognizes known individuals, and identifies smiles in real time.

## Installation

### Prerequisites

Ensure that you have the following installed:

- **Python**: Version 3.7 or above
- **pip**: Python package manager

### Step-by-Step Guide

1. **Clone the Repository**

   ```bash
   git clone https://github.com/kilofrakh/face-recognition-app.git
   cd face-recognition-app
   ```

2. **Create a Virtual Environment** (Recommended)

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   Install the required dependencies listed in the `requirements.txt` file:

   ```bash
   pip install -r requirements.txt
   ```

4. **Download Haar Cascades**

   Download the Haar cascade files for face and smile detection and place them in the `data/` directory. The files needed are:

   - `haarcascade_frontalface_default.xml`
   - `haarcascade_smile.xml`

   You can download these files from the [OpenCV GitHub repository](https://github.com/opencv/opencv/tree/master/data/haarcascades).

5. **Run the Application**

   Start the Flask server:

   ```bash
   python app.py
   ```

   Open your browser and navigate to `http://127.0.0.1:5000` to access the application.

## Usage

1. **Home Page**: The main page with navigation options for collecting face data and recognizing faces.

2. **Collect Faces**: Allows you to enter a name and start capturing facial data. The webcam will capture images, and once enough samples are collected, the data is saved for recognition.

3. **Recognize Faces**: Opens the live video feed with face recognition enabled. If a recognized face is detected, the name will be displayed along with feedback about smiling.

## Technologies Used

- **Flask**: Provides the web framework for building the application interface.
- **OpenCV**: Handles image processing tasks such as face and smile detection.
- **scikit-learn**: Used for implementing the KNN algorithm for face recognition.
- **NumPy**: A fundamental package for numerical computations in Python, used here to handle image data.
- **Bootstrap**: Used for styling the web interface, making it responsive and visually appealing.

## How It Works

### Face Detection

The application uses OpenCV’s Haar cascades for face detection. These are pre-trained classifiers that identify facial features based on patterns.

### Face Recognition

The collected facial data is used to train a KNN classifier. The classifier is trained with images labeled by the user’s name, allowing it to recognize known faces when they appear in the video feed.

### Smile Detection

Once a face is detected, another Haar cascade is used to identify smiles within the detected region. If a smile is detected, the application displays a positive message; otherwise, it encourages the user to smile.

## File Structure

```plaintext
face-recognition-app/
│
├── app.py                  # Main application script
├── templates/              # HTML templates for Flask
│   ├── index.html          # Home page
│   ├── collect_faces.html  # Face collection page
│   └── recognize_faces.html# Face recognition page
│
├── static/                 # Static files (CSS, JS)
│
├── data/                   # Directory to store data files
│   ├── haarcascade_frontalface_default.xml
│   ├── haarcascade_smile.xml
│   ├── faces_data.pkl      # Stored facial data
│   └── names.pkl           # Stored names
│
└── requirements.txt        # Required Python packages
```

## Troubleshooting

- **Webcam Not Detected**: Ensure that your webcam is properly connected and accessible. Restart your system if necessary.
- **Face Not Recognized**: Make sure you have collected sufficient face data. If recognition fails, try retraining with more samples.
- **Dependencies Issues**: Ensure that all dependencies are correctly installed as per the `requirements.txt` file.

## Contributing

We welcome contributions from the community! To contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature-branch`
3. Make your changes and commit them: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature-branch`
5. Create a pull request.

Please ensure that your code follows the project's coding standards and is well-documented.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Acknowledgements

- [OpenCV](https://opencv.org/) for providing excellent computer vision tools.
- [Flask](https://flask.palletsprojects.com/) for the lightweight web framework.
- [scikit-learn](https://scikit-learn.org/) for machine learning functionalities.
- Special thanks to the open-source community for their valuable contributions.

---

This `README.md` provides a comprehensive guide to setting up, running, and understanding the Face Recognition and Smile Detection Web Application. Customize it as needed to fit your specific project and share it to help others easily use and contribute to your application!