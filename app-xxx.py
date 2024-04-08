#save to csv file 
import os
from PIL import Image
import csv
file_prompt="book"
folder_name="images"
# Directory containing the PNG files
input_dir = folder_name
# Output file path
output_file = "sdmeta.csv"

# Create and open the CSV file in write mode
with open(output_file, 'w', newline='') as csvfile:
    # Create a CSV writer object
    csvwriter = csv.writer(csvfile)

    # Write the header row
    csvwriter.writerow(["File Name", "Positive Prompt", "Negative Prompt", "Aspect Ratio", "Width (pixels)", "Height (pixels)", "Model Configurations"])

    # Iterate over each file in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".png"):
            file_path = os.path.join(input_dir, filename)
            try:
                # Open the image file
                image = Image.open(file_path)

                # Get the image dimensions in pixels
                width_pixels, height_pixels = image.size

                # Calculate the aspect ratio to maintain proportions
                aspect_ratio = width_pixels / height_pixels

                # Get the metadata (EXIF) from the image
                metadata = image.info

                # Extract positive prompt and negative prompt
                positive_prompt = metadata.get("parameters", "").split("\n")[0]
                negative_prompt_line = metadata["parameters"].split("\n")[1]
                if ": " in negative_prompt_line:
                    negative_prompt = negative_prompt_line.split(": ")[1]
                else:
                    negative_prompt = "No negative prompt"

                # Find the index of the first occurrence of "Steps" in the parameters
                steps_index = metadata.get("parameters", "").find("Steps:")

                # Extract the information from the first "Steps" onwards
                steps_info = metadata.get("parameters", "")[steps_index:].split("\nTemplate")[0]

                # Format the steps_info into separate key-value pairs
                steps_list = ", ".join(steps_info.split(", "))

                # Write the data to the CSV file
                csvwriter.writerow([filename, positive_prompt, negative_prompt, aspect_ratio, width_pixels, height_pixels, steps_list])

                # Close the image file
                image.close()
            except Exception as e:
                print(f"Error processing file: {filename}")
                print(e)
#read image path from folder and from csv file 
import csv
from pathlib import Path
from PIL import Image
import glob
import base64
import csv
import hashlib
import chromadb
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
import os
from pathlib import Path
import google.generativeai as genai

# Set your API key
GOOGLE_API_KEY = "AIzaSyCNIQCY0yxjqW8pCaYEFAilj2TyO-t5p6I"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
# Set up the model
generation_config = {
    "temperature": 0.4,
    "top_p": 1,
    "top_k": 32,
    "max_output_tokens": 4096,}
safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "LOW"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "LOW"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "LOW"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "LOW"},]
model = genai.GenerativeModel(
    model_name="gemini-pro-vision",
    generation_config=generation_config,
    safety_settings=safety_settings
)
image_folder_path = "images"
images = glob.glob("images/*.jpg")
embeddings=[]
all_embeddings = []
model_emb = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
chroma_client = chromadb.PersistentClient(path="images.chromadb_SD")
collection = chroma_client.create_collection(name="images")
i=0
# Define the folder containing the images
image_folder_path = "images"

# Define the path to the CSV file
csv_file_path = output_file

# Open the CSV file in read mode
with open(csv_file_path, 'r') as csvfile:
    # Create a CSV reader object
    csvreader = csv.reader(csvfile)
    
    # Skip the header row
    next(csvreader)
    
    # Iterate over each row in the CSV file
    for row in csvreader:
        # Extract the image file name from the CSV row
        image_file_name = row[0]  # Assuming the image file name is in the first column
        positive_prompt=row[1]
        print(positive_prompt)
        model_config=row[6]
        print(model_config)
        # Construct the full path to the image file
        image_path = Path(image_folder_path) / image_file_name
        
        try:
            # Validate that the image exists
            if not image_path.exists():
                print(f"Could not find image: {image_path}")
                continue
            
            # Open the image file
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                
                # Process the image data as needed
                # For example, you can create a PIL Image object
                #image = Image.open(image_path)
                image_parts = [{"mime_type": "image/jpeg", "data": image_data}]

                # Do something with the image...
                response = model.generate_content(["tell me about image in short words without comma between words", image_parts[0]])
                print(response.text)

                embedding = model_emb.embed_query(response.text)
                file_hash = hashlib.sha256(image_data).digest()
                file_id = base64.b64encode(file_hash).decode()
        
                embeddings.append({
                    "positive":positive_prompt,
                    "prompts":response.text,
                    "embedding": embedding,
                    "filePath": str(image_path),  # Convert Path object to string
                    "id": file_id
                })
                
                i=i+1
                print(f"processing... {i}")
                
        except Exception as e:
            print(f"Error processing image: {image_path}")
            print(e)
        continue
# Append all embeddings to the list
all_embeddings.extend(embeddings)
csv_file_path = "embeddings.csv"
# Write all embeddings to the CSV file
with open(csv_file_path, "w", newline="", encoding="utf-8") as csv_file:

    fieldnames = ["positive","prompts","embedding", "filePath", "id"]
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    writer.writeheader()
    for embedding_data in all_embeddings:
        writer.writerow(embedding_data)
print("All embeddings saved to CSV file.")

collection.add(
    embeddings=[e["embedding"] for e in embeddings],
    #metadatas=[{"filePath": e["filePath"]},{"prompt": e["prompts"]} for e in embeddings],
    metadatas=[{"filePath": e["filePath"], "prompt": e["prompts"],"positive":e["positive"]} for e in embeddings],

    ids=[e["id"] for e in embeddings],
    
)
query_embeddings = model_emb.embed_query("woman")
results = collection.query(
    query_embeddings=[query_embeddings],
    n_results=3
)
print(results)
