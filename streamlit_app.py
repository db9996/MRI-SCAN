import streamlit as st
import requests
from PIL import Image
import io

# Title of the app
st.title("Brain MRI Segmentation")

# File uploader
uploaded_file = st.file_uploader("Upload a Brain MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Make a prediction
    if st.button("Segment"):
        # Send the image to the FastAPI server for prediction
        files = {'file': uploaded_file.getvalue()}
        try:
            response = requests.post("http://127.0.0.1:8000/predict", files=files)
            response.raise_for_status()  # Raise an error for bad responses
            result_image = Image.open(io.BytesIO(response.json()['predicted_mask']))
            st.image(result_image, caption='Predicted Mask', use_column_width=True)
        except requests.exceptions.HTTPError as e:
            st.error(f"Error in prediction: {e.response.json()['detail']}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Display instructions
st.markdown("""
### Instructions:
1. Upload a brain MRI image in JPEG or PNG format.
2. Click the "Segment" button to see the predicted segmentation mask.
""")