import os
import cv2
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from sklearn.metrics.pairwise import cosine_similarity

# Define paths
dataset_folder = 'dataset'
upload_folder = 'uploads'

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = upload_folder

# Global variables
dataset_embeddings = np.array([])
labels = np.array([])

def load_datasets():
    global dataset_embeddings, labels
    if os.path.exists('dataset_embeddings.npy') and os.path.exists('labels.npy'):
        dataset_embeddings = np.load('dataset_embeddings.npy')
        labels = np.load('labels.npy')
        if dataset_embeddings.size == 0 or labels.size == 0:
            print("Loaded files but they are empty.")
            dataset_embeddings = np.array([])
            labels = np.array([])
        else:
            print(f"Loaded dataset embeddings with shape: {dataset_embeddings.shape}")
            print(f"Loaded labels with shape: {labels.shape}")
    else:
        print("No dataset embeddings or labels found.")
        dataset_embeddings = np.array([])
        labels = np.array([])

# Load datasets
load_datasets()

class FaceEmbedder:
    def __init__(self):
        # Initialize your actual face embedding model here
        pass

    def embeddings(self, face_array):
        # Replace this with actual embedding logic
        return np.random.rand(128)  # Placeholder for the actual model's output

embedder = FaceEmbedder()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def detect_and_preprocess_face(file_path):
    image = cv2.imread(file_path)
    if image is None:
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None
    (x, y, w, h) = faces[0]
    face = image[y:y + h, x:x + w]
    face_resized = cv2.resize(face, (160, 160))
    face_resized = face_resized.astype('float32') / 255.0
    face_array = np.expand_dims(face_resized, axis=0)
    return face_array

def process_dataset(dataset_folder):
    embeddings_list = []
    labels_list = []
    for person_folder in os.listdir(dataset_folder):
        person_path = os.path.join(dataset_folder, person_folder)
        if not os.path.isdir(person_path):
            continue
        for image_file in os.listdir(person_path):
            image_path = os.path.join(person_path, image_file)
            face_array = detect_and_preprocess_face(image_path)
            if face_array is not None:
                embedding = embedder.embeddings(face_array)
                embeddings_list.append(embedding)
                labels_list.append(person_folder)
    if embeddings_list:
        embeddings_array = np.array(embeddings_list)
        labels_array = np.array(labels_list)
        np.save('dataset_embeddings.npy', embeddings_array)
        np.save('labels.npy', labels_array)
        print("Dataset processed and embeddings saved.")
    else:
        print("No faces detected in any images. Check your dataset.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    capture_option = request.form.get('capture_option')
    top_similar_faces = []

    if capture_option == 'upload':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
        else:
            return redirect(url_for('index'))
    elif capture_option == 'camera':
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        if ret:
            filename = 'captured_face.jpg'
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            cv2.imwrite(file_path, frame)
        cap.release()
    else:
        return redirect(url_for('index'))

    try:
        face_array = detect_and_preprocess_face(file_path)
        if face_array is None:
            return "No face detected. Try another image."

        captured_embedding = embedder.embeddings(face_array)
        captured_embedding = np.asarray(captured_embedding).reshape(1, -1)

        global dataset_embeddings, labels
        if dataset_embeddings.size == 0:
            num_folders = len([f for f in os.listdir(dataset_folder) if os.path.isdir(os.path.join(dataset_folder, f))])
            return f"No face embeddings in the dataset. Dataset folder contains {num_folders} folders."

        if len(dataset_embeddings.shape) == 1:
            dataset_embeddings = dataset_embeddings.reshape(1, -1)

        similarities = cosine_similarity(captured_embedding, dataset_embeddings)
        print(f'Similarity Scores: {similarities.flatten()}')

        similarity_threshold = 0.7

        for _ in range(2):
            if np.max(similarities) >= similarity_threshold:
                most_similar_index = np.argmax(similarities)
                most_similar_label = labels[most_similar_index]
                print(f'Most Similar Face: {most_similar_label}')
                top_similar_faces.append(most_similar_label)
                similarities[0, most_similar_index] = -1
            else:
                break

        print(f'Two Most Similar Faces: {top_similar_faces}')
        return render_template('result.html', top_similar_faces=top_similar_faces)

    except Exception as e:
        return f"An error occurred: {str(e)}"

if __name__ == '__main__':
    if not os.path.exists(upload_folder):
        os.makedirs(upload_folder)

    # Uncomment the line below to process the dataset (only needed once)
    # process_dataset(dataset_folder)

    app.run(host='0.0.0.0', port=5000, debug=True)
