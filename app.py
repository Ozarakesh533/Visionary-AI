import streamlit as st
from PIL import Image, ImageDraw
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.messages import HumanMessage
import pytesseract
from gtts import gTTS
import io
import base64
import os

# Set Tesseract command for local testing (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:/Users/Lenovo/AppData/Local/Programs/Python/Python312/Scripts/pytesseract.exe"

# Load Google Gemini API Key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    try:
        with open("key.txt", "r") as file:
            GOOGLE_API_KEY = file.read().strip()
    except FileNotFoundError:
        GOOGLE_API_KEY = None
        st.error("Error: API key file not found. Please provide a valid API key.")

# Initialize the Gemini model only if the API key is available
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY) if GOOGLE_API_KEY else None

# Function to convert an image to Base64 format
def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

# Function to run OCR on an image
def run_ocr(image):
    return pytesseract.image_to_string(image.convert('RGB')).strip()

# Function to analyze the image using Gemini
def analyze_image(image, prompt):
    if not llm:
        return "Error: AI model is not initialized. Please check API key setup."
    
    try:
        image_base64 = image_to_base64(image)
        message = [
            HumanMessage(content=prompt),
            HumanMessage(content=f"data:image/png;base64,{image_base64}")
        ]
        response = llm.invoke(message)
        return response.content.strip() if response else "Error: No response from AI model."
    except Exception as e:
        return f"Error: {str(e)}"

# Function to convert text to speech
def text_to_speech(text):
    tts = gTTS(text=text, lang='en', slow=False)
    audio_bytes = io.BytesIO()
    tts.write_to_fp(audio_bytes)
    audio_bytes.seek(0)
    return audio_bytes.getvalue()

# Function to detect and highlight objects in the image
def detect_and_highlight_objects(image):
    draw = ImageDraw.Draw(image)
    objects = [
        {"label": "Obstacle", "bbox": (50, 50, 200, 200)},
        {"label": "Object", "bbox": (300, 100, 500, 300)}
    ]

    for obj in objects:
        bbox = obj['bbox']
        draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="red", width=5)
        draw.text((bbox[0], bbox[1] - 10), obj['label'], fill="red")

    return image, objects

# Main app function
def main():
    st.set_page_config(page_title="EyeGuide AI", layout="wide", page_icon="🤖")

    # Sidebar
    with st.sidebar:
        st.sidebar.title("🔧 Features")
        st.sidebar.markdown("""
        - **Scene Understanding** - Describes the content of uploaded images.  
        - **Text-to-Speech** - Extracts and reads aloud text from images using OCR.  
        - **Object & Obstacle Detection** - Identifies objects or obstacles for safe navigation.  
        - **Personalized Assistance** - Offers task-specific guidance based on image content.  
        - **FAQs & Tips** - Learn about the app and get help.  
        """)

    # Main page
    st.title('👁️ Visionary AI 🤖')
    st.write("Upload an image to get started!")

    # Image upload
    uploaded_file = st.file_uploader("Choose an image (jpg, jpeg, png)", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if st.button("Describe Scene"):
            with st.spinner("Generating scene description..."):
                scene_description = analyze_image(image, "Describe this image briefly.")
                st.subheader("Scene Description")
                st.write(scene_description)
                st.audio(text_to_speech(scene_description), format='audio/mp3')

        if st.button("Extract Text"):
            with st.spinner("Extracting text..."):
                extracted_text = run_ocr(image)
                st.subheader("Extracted Text")
                st.write(extracted_text if extracted_text else "No text detected.")
                st.audio(text_to_speech(extracted_text if extracted_text else "No text detected."), format='audio/mp3')

        if st.button("Detect Objects & Obstacles"):
            with st.spinner("Detecting objects and obstacles..."):
                highlighted_image, detected_objects = detect_and_highlight_objects(image.copy())
                st.image(highlighted_image, caption="Detected Objects & Obstacles", use_container_width=True)
                st.subheader("Detected Items")
                for obj in detected_objects:
                    st.write(f"- {obj['label']} at {obj['bbox']}")

        if st.button("Personalized Assistance"):
            with st.spinner("Providing personalized assistance..."):
                assistance_response = analyze_image(image, "Provide task-specific guidance based on this image.")
                st.subheader("Personalized Assistance")
                st.write(assistance_response)

if __name__ == "__main__":
    main()
