import os
import io
from google.cloud import vision
from google.cloud.vision_v1 import types

# Set the path to your service account credentials JSON file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'Service_account_token.json'

# Initialize the Vision API client
client = vision.ImageAnnotatorClient()

def propertiesDetection(img_folder):
    # Create a results folder inside the image folder if it doesn't exist
    result_folder = os.path.join(img_folder, 'results')
    os.makedirs(result_folder, exist_ok=True)

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
                # Perform properties detection on the image
                response = client.image_properties(image=image).image_properties_annotation
                dominant_colors = response.dominant_colors

                # Create a text file to store the detection results
                result_file_path = os.path.join(result_folder, f'{os.path.splitext(file_name)[0]}.txt')
                with open(result_file_path, 'w') as result_file:
                    for color_index, color in enumerate(dominant_colors.colors, start=1):
                        result_file.write(f'Color {color_index}:\n')
                        result_file.write(f'\tPixel Fraction: {color.pixel_fraction}\n')
                        result_file.write(f'\tScore: {color.score}\n')
                        result_file.write(f'\tRed: {color.color.red}\n')
                        result_file.write(f'\tGreen: {color.color.green}\n')
                        result_file.write(f'\tBlue: {color.color.blue}\n\n')

                print(f'Detection results saved for {file_name}')
                
                # Check for any errors in the response
                if hasattr(response, 'error') and response.error.message:
                    print(f"Error for {file_name}: {response.error.message}")
            except Exception as e:
                print(f"An error occurred for {file_name}: {e}")

# Define the path to the folder containing images
img_folder = os.path.abspath("images for properties detection")

# Call the propertiesDetection function with the image folder path
propertiesDetection(img_folder)
