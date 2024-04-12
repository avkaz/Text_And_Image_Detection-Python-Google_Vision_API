import os
import io
from google.cloud import vision
from google.cloud.vision_v1 import types

# Set the path to your service account credentials JSON file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'Service_account_token.json'

# Initialize the Vision API client
client = vision.ImageAnnotatorClient()


def faceDetection(img_folder):
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
                # Perform face detection on the image
                response = client.face_detection(image=image)

                # Create a text file to store the detection results
                result_file_path = os.path.join(result_folder, f'{os.path.splitext(file_name)[0]}.txt')
                with open(result_file_path, 'w') as result_file:
                    faceAnnotations = response.face_annotations
                    likehood = ('Unknown', 'Very Unlikely', 'Unlikely', 'Possibly', 'Likely', 'Very Likely')

                    result_file.write('Faces:\n')
                    for face_index, face in enumerate(faceAnnotations, start=1):
                        result_file.write(f'Face {face_index}:\n')
                        result_file.write(f'Detection Confidence: {face.detection_confidence}\n')
                        result_file.write(f'Angry likelihood: {likehood[face.anger_likelihood]}\n')
                        result_file.write(f'Joy likelihood: {likehood[face.joy_likelihood]}\n')
                        result_file.write(f'Sorrow likelihood: {likehood[face.sorrow_likelihood]}\n')
                        result_file.write(f'Surprised likelihood: {likehood[face.surprise_likelihood]}\n')
                        result_file.write(f'Headwear likelihood: {likehood[face.headwear_likelihood]}\n')

                        face_vertices = [f'({vertex.x},{vertex.y})' for vertex in face.bounding_poly.vertices]
                        result_file.write(f'Face bounding box: {" ".join(face_vertices)}\n')

                print(f'Detection results saved for {file_name}')
                
                # Check for any errors in the response
                if response.error.message:
                    print(f"Error for {file_name}: {response.error.message}")
            except Exception as e:
                print(f"An error occurred for {file_name}: {e}")

# Define the path to the folder containing images
img_folder = os.path.abspath("images_with_faces")

# Call the faceDetection function with the image folder path
faceDetection(img_folder)
