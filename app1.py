from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Load the saved model
loaded_model = tf.keras.models.load_model('traffic_sign_model.keras')

# Get input shape from model (ignoring batch dimension = None)
_, img_height, img_width, img_channels = loaded_model.input_shape

# Scaling function that adapts channels
def scaling(image):
    # Convert based on expected channels
    if img_channels == 1:
        image = image.convert("L")  # grayscale
    else:
        image = image.convert("RGB")  # color
    
    # Resize to model expected dimensions
    img = Image.open(uploaded_file).convert("RGB").resize((50, 50))  # force RGB
    arr = np.array(img) / 255.0
    reshaped_image = arr.reshape(1, 50, 50, 3)  # batch dimension added


# Define the class labels
classes = {
    0:'Speed limit (20km/h)',
    1:'Speed limit (30km/h)',
    2:'Speed limit (50km/h)',
    3:'Speed limit (60km/h)',
    4:'Speed limit (70km/h)',
    5:'Speed limit (80km/h)',
    6:'End of speed limit (80km/h)',
    7:'Speed limit (100km/h)',
    8:'Speed limit (120km/h)',
    9:'No passing',
    10:'No passing for vehicles over 3.5 metric tons',
    11:'Right-of-way at the next intersection',
    12:'Priority road',
    13:'Yield',
    14:'Stop',
    15:'No vehicles',
    16:'Vehicles over 3.5 metric tons prohibited',
    17:'No entry',
    18:'General caution',
    19:'Dangerous curve to the left',
    20:'Dangerous curve to the right',
    21:'Double curve',
    22:'Bumpy road',
    23:'Slippery road',
    24:'Road narrows on the right',
    25:'Road work',
    26:'Traffic signals',
    27:'Pedestrians',
    28:'Children crossing',
    29:'Bicycles crossing',
    30:'Beware of ice/snow',
    31:'Wild animals crossing',
    32:'End of all speed and passing limits',
    33:'Turn right ahead',
    34:'Turn left ahead',
    35:'Ahead only',
    36:'Go straight or right',
    37:'Go straight or left',
    38:'Keep right',
    39:'Keep left',
    40:'Roundabout mandatory',
    41:'End of no passing',
    42:'End of no passing by vehicles over 3.5 metric'
}

# Streamlit UI
st.title("Traffic Sign Classification")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")  # ensure 3 channels
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image
    scaled_image = image.resize((50, 50))
    arr = np.array(scaled_image) / 255.0
    reshaped_image = arr.reshape(1, 50, 50, 3)  # model expects (50,50,3)

    # Predict
    prediction = loaded_model.predict(reshaped_image)
    predicted_class_index = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction) * 100
    predicted_class_label = classes[predicted_class_index]

    st.success(f"Predicted Traffic Sign: {predicted_class_label} "
               f"(Confidence: {confidence:.2f}%)")
