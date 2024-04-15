import os
import io
from google.cloud import vision
from google.cloud.vision_v1 import types

# Set the path to your service account credentials JSON file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'Service_account_token.json'

# Initialize the Vision API client
client = vision.ImageAnnotatorClient()


def safe_search_detection(img_folder):
    # Create a results folder inside the image folder if it doesn't exist
    result_folder = os.path.join(img_folder, 'results')
    os.makedirs(result_folder, exist_ok=True)

    # Iterate over all files in the folder
    for file_name in os.listdir(img_folder):
        # Check if the file is an image file
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            # Create the full path to the image file
            img_path = os.path.join(img_folder, file_name)

            # Read the image file
            with io.open(img_path, 'rb') as image_file:
                content = image_file.read()

            # Create an Image object
            image = types.Image(content=content)

            try:
                # Perform safe search detection on the image
                response = client.safe_search_detection(image=image)

                # Create a text file to store the detection results
                result_file_path = os.path.join(result_folder, f'{os.path.splitext(file_name)[0]}.txt')
                with open(result_file_path, 'w') as result_file:
                    safe_search_annotation = response.safe_search_annotation
                    likelihood = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE', 'LIKELY', 'VERY_LIKELY')

                    result_file.write('Safe Search Detection Results:\n')
                    result_file.write(f'adult: {likelihood[safe_search_annotation.adult]}\n')
                    result_file.write(f'spoof: {likelihood[safe_search_annotation.spoof]}\n')
                    result_file.write(f'medical: {likelihood[safe_search_annotation.medical]}\n')
                    result_file.write(f'violence: {likelihood[safe_search_annotation.violence]}\n')
                    result_file.write(f'racy: {likelihood[safe_search_annotation.racy]}\n')

                print(f'Detection results saved for {file_name}')
                
                # Check for any errors in the response
                if response.error.message:
                    print(f"Error for {file_name}: {response.error.message}")
            except Exception as e:
                print(f"An error occurred for {file_name}: {e}")

# Define the path to the folder containing images
img_folder = os.path.abspath("images for explicit content detection")

# Call the safe_search_detection function with the image folder path
safe_search_detection(img_folder)
