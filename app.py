import streamlit as st
from fastai.vision.all import *
from pathlib import Path
import io
import time

# model class
class KidneyNetV(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = models.mobilenet_v2(pretrained=True)
        self.base_model.classifier = nn.Identity()  # Remove the original classifier
        self.custom_layers = nn.Sequential(
            nn.Dropout(p=0.5),  # Added dropout for regularization
            nn.Linear(1280, 2)  # Output size should match the number of classes
        )

    def forward(self, x):
        x = self.base_model(x)  # Get features from MobileNet
        x = x.view(x.size(0), -1)  # Flatten the output
        return self.custom_layers(x)

# Loading the model
def load_model(model_path, model_name):
    try:
        dls = ImageDataLoaders.from_folder('Dataset', train='Train', valid='Test',
                                           item_tfms=Resize(224), batch_tfms=aug_transforms())
        model = KidneyNetV()  # Ensure this matches your model definition
        learn = Learner(dls, model, metrics=accuracy)
        learn.load(str(model_path / model_name))  # Updated to include the full path
        return learn
    except FileNotFoundError as e:
        st.error(f"Error: Model file not found - {e}")
    except Exception as e:
        st.error(f"Error loading model: {e}")

# Function to predict kidney stones in an image
def predict_image(learn, img):
    try:
        pred, pred_idx, probs = learn.predict(img)
        return pred, pred_idx, probs[pred_idx].item()
    except Exception as e:
        st.error(f"Error during prediction: {e}")

# Streamlit app layout
st.title("Kidney Stone Detection")
st.write("Upload a CT image of the kidney:")

# File uploader
uploaded_file = st.file_uploader("Choose a file...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Check file size
    if uploaded_file.size > 2 * 1024 * 1024:  # 2 MB limit
        st.error("Error: File size exceeds 2 MB. Please upload a smaller file.")
    else:
        # Load and display the image
        try:
            img = PILImage.create(uploaded_file)
            st.image(img, caption='Uploaded Image.', use_column_width=True)
            st.write("")
            st.write("Classifying...")

            # Show loading indicator
            with st.spinner("Loading model and making predictions..."):
                time.sleep(1)  # Simulate loading time
                # Load the model
                model_path = Path(r'C:\Users\vishn\Desktop\B.Tech - College Chetha\Vignana Pradarshanalu\GitHub Codes\Kidney_Stone_detection\Saved_Models\models')
                model_name = 'Mark_VI_KidneyNetV'
                learn = load_model(model_path, model_name)

                if learn:  # Proceed only if the model loaded successfully
                    # Make prediction
                    prediction, pred_idx, probability = predict_image(learn, img)
                    
                    # Display prediction results
                    st.write(f'Prediction: {prediction}')
                    st.write(f'Probability: {probability:.2f}')

                    # Add explanation of the prediction
                    if prediction == 'Kidney_stone':
                        st.write("### Explanation:")
                        st.write("The model predicts the presence of a kidney stone based on the features extracted from the uploaded CT image.")
                        st.write("A high probability indicates a strong likelihood of the presence of a kidney stone.")
                    else:
                        st.write("### Explanation:")
                        st.write("The model predicts the absence of a kidney stone based on the features extracted from the uploaded CT image.")
                        st.write("A high probability indicates a strong likelihood of no kidney stone present.")

        except FileNotFoundError:
            st.error("Error: File not found. Please check the file and try again.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
