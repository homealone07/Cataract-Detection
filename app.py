import streamlit as st
from PIL import Image
import numpy as np
import plotly.graph_objects as go
#from models import predict
import cv2
import numpy as np
import pandas as pd
import cv2 as cv
import streamlit as st
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
import joblib




indextable = ['dissimilarity', 'contrast',
              'homogeneity', 'energy', 'correlation', 'Label']
obj = {
    0.0: "Normal",
    1.0: "Cataract",
    2.0: "Glaucoma",
    3.0: 'Retina Disease'
}
width, height = 400, 400
distance = 10
teta = 90

# Code to extract features from Image using Gray Level Co occurrence Image


def get_feature(matrix, name):
    feature = graycoprops(matrix, name)
    result = np.average(feature)
    return result


def preprocessingImage(image):
    test_img = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    test_img_gray = cv.cvtColor(test_img, cv.COLOR_RGB2GRAY)
    test_img_thresh = cv.adaptiveThreshold(
        test_img_gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 11, 3)

    cnts = cv.findContours(
        test_img_thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)

    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        test_img_ROI = test_img[y:y+h, x:x+w]
        break

    test_img_ROI_resize = cv.resize(test_img_ROI, (width, height))
    test_img_ROI_resize_gray = cv.cvtColor(
        test_img_ROI_resize, cv.COLOR_RGB2GRAY)

    return test_img_ROI_resize_gray


def extract(path):
    data_eye = np.zeros((5, 1))

    # path = cv.imread(path)
    img = preprocessingImage(path)

    glcm = graycomatrix(img, [distance], [teta],
                        levels=256, symmetric=True, normed=True)

    for i in range(len(indextable[:-1])):
        features = []
        feature = get_feature(glcm, indextable[i])
        features.append(feature)
        data_eye[i, 0] = features[0]
    return pd.DataFrame(np.transpose(data_eye), columns=indextable[:-1])


"""
Return predicted class with its probability
"""

model = joblib.load("model.pkl")



def predict(path):
    X = extract(path)
    y = model.predict(X)[0]
    prob = model.predict_proba(X)[0, int(y)]
    return (obj[y], prob)




google_form_link = 'https://docs.google.com/forms/d/1xKeveRFf90_wCX-tjMInFC48XmFF8HOsPSQ47ruOFk0/edit'
# Load the pre-trained Haar Cascade for eye detection
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Function to check if the image contains an eye
def contains_eye(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(eyes) > 0

# Function to load the image
def load_image(image_file):
    img = Image.open(image_file)
    return img

# Streamlit app title
st.title("Eye Cataract Detection")

# Streamlit header and subheader
st.header('Upload the image')

# File uploader widget
image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

# Check if an image file is uploaded
if image_file is not None:
    img = load_image(image_file)

    # Display the uploaded image
    st.image(img, width=250)

    # Convert image to OpenCV format
    open_cv_image = np.array(img)

    if contains_eye(open_cv_image):
        # Button for detection
        if st.button('Detect'):  # This line adds a 'Detect' button
            # Predict the label and probability
            label, prob = predict(open_cv_image)

            # Use markdown to style the text and include emojis
            if prob > 0.5:
                st.markdown(f"<h2 style='color: red;'>Cataract Detected 😟</h2>", unsafe_allow_html=True)
                #st.markdown(f"### Probability: **{prob:.2f}**")
            else:
                st.markdown(f"<h2 style='color: green;'>No Cataract Detected 😄</h2>", unsafe_allow_html=True)
                #st.markdown(f"### Probability: **{prob:.2f}**")

            # Pie chart visualization
            fig = go.Figure(data=[go.Pie(labels=['Cataract', 'No Cataract'], 
                                        values=[prob, 1 - prob],
                                        hoverinfo='label+percent', 
                                        pull=[0, 0])])
            fig.update_layout(title_text='Cataract Detection Probability')
            st.plotly_chart(fig)

            st.subheader("Doctor's Verification")
            st.markdown(f"[Click here to provide feedback on the cataract detection results]({google_form_link})", unsafe_allow_html=True)

    else:
        st.error("No eyes detected in the image. Please upload a relevant eye image.")
