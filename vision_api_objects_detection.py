import os
import io
from numpy import random
from google.cloud import vision
from google.cloud.vision_v1 import types
from Pillow_Utility import draw_borders, Image
import pandas as pd

# Set the path to your service account credentials JSON file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'Service_account_token.json'

# Initialize the Vision API client
client = vision.ImageAnnotatorClient()


def objectsDetection(img_folder):
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
                # Perform object detection on the image
                response = client.object_localization(image=image)
                localized_object_annotations = response.localized_object_annotations

                # Create lists to store the detection results
                names = []
                scores = []
                for obj in localized_object_annotations:
                    names.append(obj.name)
                    scores.append(obj.score)

                # Create a DataFrame from the lists
                df = pd.DataFrame({'Name': names, 'Score': scores})

                # Open the image with PIL
                pillow_image = Image.open(img_path)

                # Draw bounding boxes and labels on the image
                for obj in localized_object_annotations:
                    r, g, b = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
                    pillow_image = draw_borders(pillow_image, obj.bounding_poly, (r, g, b),
                            caption=f'{obj.name} - {obj.score * 100:.2f}%',
                            confidence_score=obj.score,
                            border_width=5)  # Increase border width to 5 pixels


                # Show the image
                pillow_image.show()

                # Save the detection results to a CSV file
                result_file_path = os.path.join(result_folder, f'{os.path.splitext(file_name)[0]}.csv')
                df.to_csv(result_file_path, index=False)
                print(f'Detection results saved for {file_name}')

                # Check for any errors in the response
                if response.error.message:
                    print(f"Error for {file_name}: {response.error.message}")

            except Exception as e:
                print(f"An error occurred for {file_name}: {e}")

# Define the path to the folder containing images
img_folder = os.path.abspath("images for objects detection")

# Call the objectsDetection function with the image folder path
objectsDetection(img_folder)