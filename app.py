import streamlit as st
import cv2
import easyocr
import re
import numpy as np

def enhance_face_image(face_image):
    # Convert the face image to grayscale
    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

    # Apply histogram equalization to enhance the contrast
    enhanced_face = cv2.equalizeHist(gray_face)

    return enhanced_face

def extract_text_and_face(image):
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

    # Initialize a list to store the extracted text
    extracted_text = []

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

                # Enhance the face image
                enhanced_face = enhance_face_image(face_image)

                # Display the original and enhanced face images
                st.image([face_image, enhanced_face], caption=['Original Face', 'Enhanced Face'], use_column_width=True)

                face_detected = True  # Set to True after the first face is detected

        for detection in result:
            extracted_text.append(detection[1])  # Extracted text
            st.text(detection[1])

    # Convert the extracted text list into a single string
    text = '\n'.join(extracted_text)

    # Your existing code for processing the text
    res = text.split()
    name = None
    dob = None
    adh = None
    sex = None
    nameline = []
    dobline = []
    text0 = []
    text1 = []
    text2 = []
    lines = text.split('\n')
    for lin in lines:
        s = lin.strip()
        s = lin.replace('\n', '')
        s = s.rstrip()
        s = s.lstrip()
        text1.append(s)

    if 'female' in text.lower():
        sex = "FEMALE"
    else:
        sex = "MALE"

    text1 = list(filter(None, text1))
    text0 = text1[:]

    try:
        # Cleaning first names
        name = text0[0]
        name = name.rstrip()
        name = name.lstrip()
        name = name.replace("8", "B")
        name = name.replace("0", "D")
        name = name.replace("6", "G")
        name = name.replace("1", "I")
        name = re.sub('[^a-zA-Z]+', ' ', name)

        # Cleaning DOB
        dob = text0[1][-10:]
        dob = dob.rstrip()
        dob = dob.lstrip()
        dob = dob.replace('l', '/')
        dob = dob.replace('L', '/')
        dob = dob.replace('I', '/')
        dob = dob.replace('i', '/')
        dob = dob.replace('|', '/')
        dob = dob.replace('\"', '/1')
        dob = dob.replace(":", "")
        dob = dob.replace(" ", "")

        # Cleaning Aadhar number details
        aadhar_number = ''.join(filter(str.isdigit, res[0]))

        if len(aadhar_number) >= 10:
            st.write("Aadhar number is: " + aadhar_number)
        else:
            st.write("Aadhar number not read")
        adh = aadhar_number

    except:
        pass

# Streamlit UI
st.title("Text and Face Extraction App")
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Read the image using OpenCV
    image = cv2.imdecode(np.frombuffer(uploaded_image.read(), np.uint8), 1)
    extract_text_and_face(image)
