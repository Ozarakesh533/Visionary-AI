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

# Set Tesseract command for local testing (Windows)
pytesseract.pytesseract.tesseract_cmd = r"C:/Users/Lenovo/AppData/Local/Programs/Python/Python312/Scripts/pytesseract.exe"

# Configure Google Gemini API Key
GOOGLE_API_KEY = "key.txt"  # Add your API key
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

# Function to convert an image to Base64 format
def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

# Function to run OCR on an image
def run_ocr(image):
    return pytesseract.image_to_string(image).strip()

# Function to analyze the image using Gemini
def analyze_image(image, prompt):
    try:
        image_base64 = image_to_base64(image)
        message = HumanMessage(
            content=[ 
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": f"data:image/png;base64,{image_base64}"}
            ]
        )
        response = llm.invoke([message])
        return response.content.strip()
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

    # Custom CSS for sidebar, main page, and full-width text
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;700&display=swap');

        /* Sidebar styling */
        [data-testid="stSidebar"] {
            background-color: #e0f7fa; /* Light cyan */
            color: black;
        }
        [data-testid="stSidebar"] a {
            color: blue !important;
            text-decoration: none;
        }
        [data-testid="stSidebar"] a:hover {
            text-decoration: underline;
        }

        /* Full-width text styling */
        .full-width-text { 
            width: 100%;
            font-family: 'JetBrains Mono', monospace; 
            font-size: 16px; 
            word-wrap: break-word;
        }

        /* Header styling */
        h1 {
            color: #00796b;
            font-family: 'JetBrains Mono', monospace;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

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
        
        # FAQs Section
        if st.button("❓ FAQs"):
            st.markdown("""
            **Q1:** How does this app work?  
            A1: Upload an image, and the app uses AI for scene understanding, text-to-speech conversion, and object detection.  

            **Q2:** What is OCR?  
            A2: OCR (Optical Character Recognition) is the process of extracting text from images.  

            **Q3:** Is the app secure?  
            A3: Yes, the app does not store any uploaded data.  
            """)

        # Developer Information
        if st.button("🔗 About Developer"):
            st.markdown("""
            **Name:** Rakesh Oza  
            **GitHub:** [github.com/rakeshoza](https://github.com/rakeshoza)  
            **LinkedIn:** [linkedin.com/in/rakesh-oza](https://www.linkedin.com/in/rakesh-oza)  
            """)
            st.markdown("---")

    # Main page
    st.title('👁️ Visionary AI 🤖')
    st.write("""  
        This tool assists visually impaired individuals by leveraging image analysis.  
        It provides the following features:  
        - **Scene Understanding**  
        - **Text-to-Speech Conversion**  
        - **Object & Obstacle Detection**  
        - **Personalized Assistance**  
        Upload an image to get started and let AI help you understand and interact with your environment!  
    """)

    # Image upload
    st.header("📂 Drop Image Below")
    uploaded_file = st.file_uploader("Choose an image (jpg, jpeg, png)", type=['jpg', 'jpeg', 'png'])

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Action buttons
        if st.button("Describe Scene"):
            with st.spinner("Generating scene description..."):
                scene_prompt = "Describe this image briefly."
                scene_description = analyze_image(image, scene_prompt)
                st.subheader("Scene Description")
                st.markdown(
                    f"<div class='full-width-text'>{scene_description}</div>", unsafe_allow_html=True
                )
                st.audio(text_to_speech(scene_description), format='audio/mp3')

        if st.button("Extract Text"):
            with st.spinner("Extracting text..."):
                extracted_text = run_ocr(image)
                st.subheader("Extracted Text")
                if extracted_text:
                    st.markdown(
                        f"<div class='full-width-text'>{extracted_text}</div>", unsafe_allow_html=True
                    )
                    st.audio(text_to_speech(extracted_text), format='audio/mp3')
                else:
                    no_text_message = "No text detected in the image."
                    st.markdown(
                        f"<div class='full-width-text'>{no_text_message}</div>", unsafe_allow_html=True
                    )
                    st.audio(text_to_speech(no_text_message), format='audio/mp3')

        if st.button("Detect Objects & Obstacles"):
            with st.spinner("Identifying objects and obstacles..."):
                obstacle_prompt = "Identify objects or obstacles in this image and provide their positions for safe navigation."
                obstacle_description = analyze_image(image, obstacle_prompt)
                st.subheader("Objects & Obstacles Detected")
                st.markdown(
                    f"<div class='full-width-text'>{obstacle_description}</div>", unsafe_allow_html=True
                )
                st.audio(text_to_speech(obstacle_description), format='audio/mp3')

        if st.button("Personalized Assistance"):
            with st.spinner("Providing personalized guidance..."):
                task_prompt = "Provide task-specific guidance based on the content of this image."
                assistance_description = analyze_image(image, task_prompt)
                st.subheader("Personalized Assistance")
                st.markdown(
                    f"<div class='full-width-text'>{assistance_description}</div>", unsafe_allow_html=True
                )
                st.audio(text_to_speech(assistance_description), format='audio/mp3')


if __name__ == "__main__":
    main()
