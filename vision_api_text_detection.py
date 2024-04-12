import os
import io
from google.cloud import vision
from google.cloud.vision_v1 import types
import pandas as pd

# Set the path to your service account credentials JSON file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'Service_account_token.json'

# Initialize the Vision API client
client = vision.ImageAnnotatorClient()

def clean_text(text):
    # Remove newlines and symbols except spaces
    cleaned_text = text.replace('\n', ' ')
    return cleaned_text


##################### detect local images #####################
def detectTextLocalImage(img_folder):
    # Initialize a list to store the results
    results = []

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
                
                # Append the result to the list
                results.append({'Image': file_name, 'Text': cleaned_text})
                
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

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(results)
    return df

# Define the path to the folder containing images
img_folder = os.path.abspath("images with text")

# Call the detectText function with the image folder path
df = detectTextLocalImage(img_folder)

# Print the DataFrame
print(df)


##################### detect images via url #####################

def detectTextURLImage(url):
    image = types.Image()
    image.source.image_uri = url
    # Initialize a list to store the results
    results = []
    try:
        # Perform text detection on the image
        response = client.text_detection(image=image)
        
        # Extract text annotations
        texts = response.text_annotations
        detected_text = texts[0].description if texts else "No text detected"
        
        # Clean the detected text
        cleaned_text = clean_text(detected_text)
        
        # Append the result to the list
        results.append({'Image': url, 'Text': cleaned_text})
        
        # Check for any errors in the response
        if response.error.message:
            print(f"Error for {url}: {response.error.message}")
        else:
            # Print the label annotations
            labels = response.label_annotations
            for label in labels:
                print(label.description)
    except Exception as e:
        print(f"An error occurred for {url}: {e}")

    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(results)
    return df

image_url = "https://images.unsplash.com/photo-1528716321680-815a8cdb8cbe?q=80&w=2565&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"

df = detectTextURLImage(image_url)

# Print the DataFrame
print(df)