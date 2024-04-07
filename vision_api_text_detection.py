import os
import io
from google.cloud import vision
from google.cloud.vision_v1 import types

# Set the path to your service account credentials JSON file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'Service_account_token.json'

# Initialize the Vision API client
client = vision.ImageAnnotatorClient()

def clean_text(text):
    # Remove newlines and symbols except spaces
    cleaned_text = text.replace('\n', ' ')
    return cleaned_text

def detectTextLocalImage(img_folder):
    # Iterate over all files in the folder
    for file_name in os.listdir(img_folder):
        # Check if the file is an image file
        if file_name.endswith(('.jpg', '.jpeg', '.png')):
            # Create the full path to the image file
            img_path = os.path.join(img_folder, file_name)

            # Read the image file
            with io.open(img_path, 'rb') as image_file:
                content = image_file.read()

            # Create an Image object
            image = types.Image(content=content)

            try:
                # Perform text detection on the image
                response = client.text_detection(image=image)
                
                # Extract text annotations
                texts = response.text_annotations
                detected_text = texts[0].description if texts else "No text detected"
                
                # Clean the detected text
                cleaned_text = clean_text(detected_text)
                
                # Write the detected text to a text file
                output_filename = os.path.splitext(file_name)[0] + ".txt"
                output_path = os.path.join(img_folder, output_filename)
                with open(output_path, 'w') as text_file:
                    text_file.write(cleaned_text)
                
                # Check for any errors in the response
                if response.error.message:
                    print(f"Error for {file_name}: {response.error.message}")
                else:
                    # Print the label annotations
                    labels = response.label_annotations
                    for label in labels:
                        print(label.description)
            except Exception as e:
                print(f"An error occurred for {file_name}: {e}")

# Define the path to the folder containing images
img_folder = os.path.abspath("images with text")

# Call the detectTextLocalImage function with the image folder path
detectTextLocalImage(img_folder)
