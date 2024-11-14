# Men's Hairstyle Recommendation Using Face Recognition

This project uses face recognition and machine learning to recommend men's hairstyles based on facial features. By analyzing face shape and hair type, the model aims to provide personalized hairstyle suggestions that suit individual characteristics.

## Table of Contents
- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Setting Up the Environment](#setting-up-the-environment)
  - [Loading and Preparing the Dataset](#loading-and-preparing-the-dataset)
  - [Data Processing](#data-processing)
  - [Feature Extraction](#feature-extraction)
  - [Model Training and Evaluation](#model-training-and-evaluation)
- [Output](#output)

## Project Overview

This project consists of two main components:
1. **Face Detection and Feature Extraction**: Uses MTCNN to detect facial features and FaceNet to create embeddings of the detected faces.
2. **Hairstyle Recommendation**: A K-Nearest Neighbors (KNN) classifier suggests hairstyles based on facial features extracted by the model.

The output includes:
- A trained machine learning model for face shape and hairstyle classification.
- Recommendations for hairstyles tailored to specific face shapes.

## Prerequisites

Ensure you have the following dependencies installed:
- Python 3.x
- OpenCV
- MTCNN
- Keras FaceNet
- Scikit-learn
- Seaborn

To install the necessary libraries, use:
```bash
pip install opencv-python-headless mtcnn keras-facenet seaborn scikit-learn
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/mens-hairstyle-recommendation.git
   cd mens-hairstyle-recommendation
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the dataset by uploading it to Google Drive or by storing it locally.

Here's the **Usage** section in the requested format:

---

## Usage

### Setting Up the Environment

1. **Install Required Libraries**: Install essential packages for the project:
   ```bash
   pip install opencv-python-headless mtcnn keras-facenet seaborn scikit-learn
   ```

### Loading and Preparing the Dataset

2. **Access Dataset from Google Drive**:
   - Mount Google Drive to access the dataset stored there:
     ```python
     from google.colab import drive
     drive.mount('/content/drive')
     ```
   - Extract the dataset from a zip file in Google Drive to a local directory:
     ```python
     import zipfile
     zip_path = '/content/drive/My Drive/Dataset Research/dataset.zip'
     extract_path = '/content/dataset_frans'
     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
         zip_ref.extractall(extract_path)
     ```

3. **Define Image Paths and Classes**:
   - Specify the directories and subfolders for images categorized by face shape, hair length, and hairstyle to enable organized data loading:
     ```python
     base_dir = '/content/dataset_frans/DatSet'
     face_shapes = ['diamond', 'oval', 'round', 'square']
     hair_lengths = ['medium hair', 'short hair']
     hair_styles = ['Curly', 'Straight', 'Wavy']
     ```

### Data Processing

4. **Load and Resize Images**:
   - Iterate through each folder, load images, and resize them to 160x160 pixels (required for FaceNet). Store each image and its corresponding labels:
     ```python
     images = []
     labels = []
     # Loop through each category and load images
     ```

5. **Convert Data to Numpy Arrays**:
   - Convert image and label lists to numpy arrays for compatibility with machine learning models:
     ```python
     images = np.array(images)
     labels = np.array(labels)
     ```

6. **Encode Labels**:
   - Use `LabelEncoder` to transform face shape, hair length, and hairstyle labels into numeric values:
     ```python
     from sklearn.preprocessing import LabelEncoder
     # Encode labels
     ```

### Feature Extraction

7. **Initialize Models for Face Detection and Embeddings**:
   - Load MTCNN for face detection and FaceNet for generating facial embeddings:
     ```python
     from mtcnn import MTCNN
     from keras_facenet import FaceNet
     detector = MTCNN()
     embedder = FaceNet()
     ```

8. **Extract Facial Embeddings**:
   - Define a function to detect faces, crop and resize them, and generate embeddings using FaceNet. Extract features for all images where face detection is successful:
     ```python
     def extract_face_embeddings(img, detector, embedder):
         # Detect and embed face
     ```

### Model Training and Evaluation

9. **Train the Model**:
   - Use extracted facial embeddings and encoded labels to train a K-Nearest Neighbors (KNN) classifier for hairstyle recommendations based on face shape:
     ```python
     from sklearn.neighbors import KNeighborsClassifier
     # Train KNN classifier
     ```

10. **Evaluate the Model**:
    - Use classification metrics like accuracy and a confusion matrix to evaluate the performance of the model on test data.

## Output

- **Face Detection and Embedding Extraction**: Generates face embeddings for each image.
- **Trained Model**: Stores a trained KNN model to recommend hairstyles.
- **Hairstyle Recommendations**: Outputs suggested hairstyles based on the provided face image.

### Example Output
```bash
Recommended Hairstyle: Short Textured Crop
```
