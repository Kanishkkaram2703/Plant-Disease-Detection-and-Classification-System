# Plant-Disease-Detection-and-Classification-System

Building a plant disease prediction model using a Convolutional Neural Network (CNN) for image classification can be a rewarding project. This involves several steps, including dataset preparation, model building, training, evaluation, and deploying the model using a Graphical User Interface (GUI) through a tool like Streamlit. I'll walk you through each step of the process, providing a deep explanation along the way.

Prerequisites:
Python: For building and running the code.
PyCharm IDE: For writing and testing your code.
Streamlit: For creating a simple and interactive web-based GUI.
TensorFlow/Keras: For building and training the CNN model.
Pillow: For image processing.
Step-by-Step Explanation
1. Dataset Collection and Preprocessing
First, you need a dataset of plant leaf images labeled with their respective diseases. You can use a dataset like the PlantVillage dataset, which has thousands of images across many plant species and their diseases.

Data Structure: The dataset should be organized in folders, where each folder corresponds to a class (e.g., healthy, powdery mildew, rust, etc.). Each folder contains images of that particular class.

Preprocessing:

Image resizing: CNN models require images to be of the same size. Common sizes include 128x128, 224x224, or 256x256 pixels.
Normalization: Pixel values are usually normalized to a range between 0 and 1.
Data Augmentation: This involves applying transformations like rotation, flipping, or zooming to artificially increase the size of the dataset.
Code Example (Preprocessing using Keras):


2. Building the CNN Model
CNNs are powerful tools for image classification as they automatically detect spatial hierarchies in images. A typical CNN consists of convolutional layers, pooling layers, and fully connected layers.

Key Layers:
Convolutional Layer: Extracts features from the image.
Pooling Layer: Reduces the spatial dimensions of the image, retaining the most important information.
Fully Connected Layer: Makes the final classification.
Sample CNN Architecture:

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

# 1st Convolutional Layer
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 2nd Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# 3rd Convolutional Layer
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the layers
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Dropout to prevent overfitting
model.add(Dense(64, activation='relu'))

# Output Layer (number of classes = n_classes)
model.add(Dense(n_classes, activation='softmax'))

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
Model Explanation:
Conv2D layers: These extract features like edges and textures from images. The filter size of (3,3) is a common choice.
MaxPooling2D: Reduces the spatial size of the representation, which makes the model more computationally efficient and helps prevent overfitting.
Dense layers: Fully connected layers, used to interpret the features and predict the output class.
Dropout layer: Helps prevent overfitting by randomly setting some of the neurons to zero during training.
3. Training the CNN Model
Once your model is built, you can start training it using the preprocessed dataset.


# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator  # A separate validation set
)

# Save the trained model
model.save('plant_disease_model.h5')
Epochs: The number of times the model will see the entire dataset. A good range is between 10 to 50, depending on the complexity of the data.
Validation Set: Helps in tuning hyperparameters and checking the model’s performance during training.
4. Model Evaluation
Once the training is done, you need to evaluate the model on a separate test dataset. This will give you an idea of how well the model generalizes to unseen data.


# Evaluate the model
test_loss, test_acc = model.evaluate(test_generator)
print(f'Test accuracy: {test_acc}')
5. Building the Streamlit GUI
Now that you have a trained model, the next step is to build a simple GUI using Streamlit. Streamlit makes it easy to deploy machine learning models on the web with very little code.

Install Streamlit:
bash
Copy code
pip install streamlit
Basic Streamlit App:
python
Copy code
import streamlit as st
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('plant_disease_model.h5')

# Title
st.title("Plant Disease Prediction")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Preprocess the image for the model
    img = image.resize((128, 128))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    
    # Make a prediction
    predictions = model.predict(img)
    score = np.argmax(predictions[0])
    
    st.write(f"Prediction: {class_names[score]}")  # class_names = list of classes
Explanation:
Streamlit Widgets: st.file_uploader allows users to upload an image. st.image displays the uploaded image.
Model Inference: The uploaded image is resized and preprocessed before being passed to the trained model. The predicted class is then displayed.
6. Running the Streamlit App
To run the app, navigate to your project directory and use the following command:

bash
Copy code
streamlit run app.py
This will open a browser window displaying the GUI, where you can upload plant leaf images and get disease predictions in real time.

Conclusion:
By following these steps, you can build a CNN-based plant disease prediction system and deploy it using Streamlit. The CNN model will learn to identify patterns in leaf images and classify them according to different plant diseases. Streamlit provides an intuitive way to present your model and interact with users, making it easy to upload images and get predictions.






