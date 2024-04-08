import streamlit as st
import chromadb
#import llm
import time
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
from pathlib import Path
import google.generativeai as genai

# Set your API key
GOOGLE_API_KEY = "AIzaSyCNIQCY0yxjqW8pCaYEFAilj2TyO-t5p6I"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

#pip install llm-clip
# Streamlit config
st.set_page_config(layout="wide")
st.title("Image search engine")

# Enter search term or provide image
option = st.selectbox('How do you want to search?', ('Search Term', 'Image'))
if option == "Search Term":
    uploaded_file = None
    search_term = st.text_input("Enter search term")
else:
    search_term = None
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        st.image(uploaded_file, width=200)

st.markdown('<hr style="border:1px #00008B; border-style: solid; margin:0;">', 
    unsafe_allow_html=True)

generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,}
safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
    
    ]



model = genai.GenerativeModel(
    model_name="gemini-pro-vision",
    generation_config=generation_config,
    safety_settings=safety_settings
)
model_emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

chroma_client = chromadb.PersistentClient(path="images.chromadb_SD")
collection = chroma_client.get_collection(name="images")

with st.empty():
    if option and (uploaded_file or search_term):
        start = time.time()
        with st.spinner('Searching'):
            if option == 'Search Term':
                #query_embeddings = model.embed(search_term)
                query_embeddings = model_emb.embed_query(search_term)
            else:
                image = Image.open(uploaded_file)
                #image_data = uploaded_file.read()
                    # Generate content for the current image
                #image_parts = [{"mime_type": "image/jpeg", "data": image_data}]
        
                response = model.generate_content(["tell me about image in short words without comma between words", image])
                query_embeddings = model_emb.embed_query(response.text)#model.embed(search_term)

                #query_embeddings = model.embed(uploaded_file.getvalue())
            result = collection.query(
                query_embeddings=[query_embeddings],
                n_results=1
            )
        end = time.time()

        metadatas = result["metadatas"][0]
        #st.markdown(metadatas)
        #print(metadatas)
        distances = result["distances"][0]
        with st.container():
            st.write(f"**Results** ({end-start:.2f} seconds)")
            for index, id in enumerate(result["ids"][0]):
                left, right = st.columns([0.5, 0.5])
                if distances[index]==0 or distances[index]<=0.3:
                    with left:
                        st.markdown(metadatas[index]['filePath'])
                        path="/opt/render/project/src/"+metadatas[index]['filePath']
                        st.markdown(path)
                        st.image(Image.open(metadatas[index]['filePath']), width=500)
                        #st.markdown(metadatas[index]['prompt'])
                       # st.markdown(metadatas[index]['positive'])
                    with right:
                        st.markdown(metadatas[index]['positive'])
                        #st.markdown(f"""**Id**: {id}  
                           # **Distance**: {distances[index]}""")
                else:
                    st.write("No results to show. Enter a search term above.")
                   # st.markdown(f"""**Id**: {id}**Distance**: {distances[index]}
                       # """)
    
    else:           
        st.write("No results to show. Enter a search term above.")
