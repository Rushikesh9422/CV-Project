import streamlit as st
import pandas as pd
import cv2
import base64
from io import BytesIO
from PIL import Image
import easyocr

def extract_text_and_face(image_path):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply thresholding to make text stand out
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Create an EasyOCR reader for English, Hindi, and Marathi scripts
    reader = easyocr.Reader(['en', 'hi', 'mr'])

    # Find text regions using contour detection
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Initialize a list to store the extracted text and face image base64 strings
    extracted_data = []

    # Initialize a variable to store the extracted text
    text = ''

    # Initialize variables for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    face_detected = False

    # Extract and process text from the identified regions
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        text_region = image[y:y + h, x:x + w]
        result = reader.readtext(text_region)

        # Check if a face is detected in the text region
        if not face_detected:
            gray_text_region = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray_text_region, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
            if len(faces) > 0:
                fx, fy, fw, fh = faces[0]
                face_image = text_region[fy:fy + fh, fx:fx + fw]

                # Save the face image to a base64 string
                _, buffer_face = cv2.imencode('.png', face_image)
                face_image_base64 = base64.b64encode(buffer_face).decode('utf-8')
                st.image(face_image, caption="Detected Face", use_column_width=True)

                face_detected = True  # Set to True after the first face is detected

        for detection in result:
            text += detection[1] + '\n'  # Concatenate the extracted text

            extracted_data.append({
                'Text': detection[1],
                'FaceImageBase64': face_image_base64,  # Save the face image base64 string
                'X': x,
                'Y': y,
                'Width': w,
                'Height': h
            })

    # Print the extracted text for reference
    st.text("Extracted Text:")
    st.text(text)

    # Read the existing Excel file if it exists
    try:
        existing_df = pd.read_excel("output.xlsx")
    except FileNotFoundError:
        # If the file doesn't exist, create a new DataFrame
        existing_df = pd.DataFrame()

    # Append the new data to the existing DataFrame
    existing_df = existing_df.append(pd.DataFrame(extracted_data), ignore_index=True)

    # Save the updated DataFrame to the Excel file
    existing_df.to_excel("output.xlsx", index=False)
    st.markdown("[Download Excel File](output.xlsx)")

# Streamlit app
st.title("Text and Face Extraction App")

# File upload
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Perform text and face extraction
    extract_text_and_face(uploaded_file.name)
