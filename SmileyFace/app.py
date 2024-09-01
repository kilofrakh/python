
from flask import Flask, render_template, Response, redirect, url_for, request
import cv2
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os

app = Flask(__name__)

# Load the pre-trained face and smile cascade classifiers
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_default.xml')
smileCascade = cv2.CascadeClassifier('data/haarcascade_smile.xml')

# Home page route
@app.route('/')
def index():
    # Render the home page
    return render_template('index.html', title="Home")

# collect face data route
@app.route('/collect_faces', methods=['GET', 'POST'])
def collect_faces():
    if request.method == 'POST':
          # Get the name from the form input
        name = request.form['name'].strip()

        # Check if the name is not empty
        if not name:
            return render_template('collect_faces.html', title="Collect Face Data", error="Name cannot be empty.")

        # Start video capture using the webcam
        video = cv2.VideoCapture(0)
        faces_data = [] # List to store flattened face images
        count = 0  # Counter for collected face samples

        while True:
            ret, frame = video.read()
            # If it doesn't return beak
            if not ret:
                break
            #Converting the video to gray fro opencv 
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       
            # Detect faces in the grayscale frame
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # Iterate over detected faces
            for (x, y, w, h) in faces:
                 # Extract the face (ROI)
                face_img = frame[y:y+h, x:x+w]
                # Resize the face to a consistent size
                resized_face = cv2.resize(face_img, (50, 50))
                # Flatten the resized face and append to faces_data
                faces_data.append(resized_face.flatten())
                # Increment the counter
                count += 1 


                # Draw a rectangle around the face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                 # Display the number of collected face samples
                cv2.putText(frame, f'Collected: {count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)  
                
                #Break if the taken pics for the face is equal to 100
                if count >= 100:
                    break


            #Click q to quit
            cv2.imshow('Collecting Faces - Press "q" to quit', frame)
            if cv2.waitKey(1) & 0xFF == ord('q') or count >= 100:
                break

        # Release the video capture and close any OpenCV windows
        video.release()
        cv2.destroyAllWindows()

        # Convert the collected face data to a NumPy array
        faces_data = np.array(faces_data)

        # saving nmes
        names_path = 'data/names.pkl'

        # Load existing names if the file exists, otherwise initialize an empty list
        if os.path.exists(names_path):
            with open(names_path, 'rb') as f:
                names = pickle.load(f)
        else:
            names = []

        # Add the collected name to the names list
        names += [name] * len(faces_data)

        # Save the updated names list
        with open(names_path, 'wb') as f:
            pickle.dump(names, f)

        # Path to save face data
        faces_data_path = 'data/faces_data.pkl'
        # Load existing face data if the file exist
        if os.path.exists(faces_data_path):
            with open(faces_data_path, 'rb') as f:
                existing_faces = pickle.load(f)
            # Append the new face data to the existing data
            faces_data = np.vstack((existing_faces, faces_data))

        # Save the updated face data
        with open(faces_data_path, 'wb') as f:
            pickle.dump(faces_data, f)

        # Redirect to the home page after face collection is complete
        return redirect(url_for('index'))

    # Render the face collection page
    return render_template('collect_faces.html', title="Collect Face Data")

# face recognition page route
@app.route('/recognize_faces')
def recognize_faces():
    # Render the face recognition page
    return render_template('recognize_faces.html', title="Recognize Faces")

# video feed
@app.route('/video_feed')
def video_feed():
    def generate_frames():
        # Start video capture using the webcam
        video = cv2.VideoCapture(0)

        # Load the saved names and face data
        names_path = 'data/names.pkl'
        faces_data_path = 'data/faces_data.pkl'

        # Check if the necessary files exist, otherwise exit
        if not os.path.exists(names_path) or not os.path.exists(faces_data_path):
            video.release()
            return

        with open(names_path, 'rb') as f:
            names = pickle.load(f)

        with open(faces_data_path, 'rb') as f:
            faces = pickle.load(f)

        # Train the KNN classifier using the collected face data
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(faces, names)

        while True:
             # Read a frame from the video capture
            success, frame = video.read()
            if not success:
                break
            
            # Convert the frame to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # Detect faces in the grayscale frame
            detected_faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            # Iterate over detected faces
            for (x, y, w, h) in detected_faces:
                # Extract and resize the face region for recognition
                face_img = frame[y:y+h, x:x+w]
                resized_face = cv2.resize(face_img, (50, 50)).flatten().reshape(1, -1)

                
                # Predict the name using the trained KNN model
                prediction = knn.predict(resized_face)
                name = prediction[0]
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                
                 # Detect smile within the face region
                roi_gray = gray[y:y+h, x:x+w]
                smiles = smileCascade.detectMultiScale(roi_gray, 1.8, 10)
                
                if len(smiles) > 0:
                    text = f"{name} - Nice Smile"
                    for (sx, sy, sw, sh) in smiles:
                        # Draw a rectangle around the detected smile
                        cv2.rectangle(frame, (x+sx, y+sy), (x+sx+sw, y+sy+sh), (0, 255, 0), 2)
                else:
                    text = f"{name} - Why not smile?"
                
                # Display the name and smile status above the face   
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2, cv2.LINE_AA)


            
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in a format for streaming
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        # Release the video capture when done
        video.release()

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # Ensure the 'data' directory exists
    if not os.path.exists('data'):
        os.makedirs('data')

    # Start the Flask debug   
    app.run(debug=True)
