import os, io
from google.cloud import vision
from google.cloud.vision_v1 import types

# Ensure 'io' is imported only if necessary
# Remove unnecessary imports
# import io
import pandas as pd

# Set the path to your service account credentials JSON file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'Service_account_token.json'

# Initialize the Vision API client
client = vision.ImageAnnotatorClient()

def cropHint(file_path, aspect_ratios):
    with io.open(file_path, 'rb') as image_file:
            content = image_file.read()

    image = types.Image(content=content)

    # desired ratio
    crop_hints_params = types.CropHintsParams(aspect_ratios=aspect_ratios)
    image_context = types.ImageContext(
        crop_hints_params=crop_hints_params)

    response = client.crop_hints(
        image=image,
        image_context=image_context
        )
    
    cropHints = response.crop_hints_annotation.crop_hints

    for cropHint in cropHints:
        print('Confidence:', cropHint.confidence)
        print('Importance Fraction:', cropHint.importance_fraction)
        print('Vertices:', cropHint.bounding_poly.vertices)


image_path = os.path.abspath('images_for_image_understanding/pexels-joan-costa-17733685.jpg')
cropHint(image_path, [16/9])
