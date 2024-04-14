import os
import io
from google.cloud import vision
from google.cloud.vision_v1 import types
import pandas as pd
from draw_vertice import drawVertices 

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'Service_account_token.json'
client = vision.ImageAnnotatorClient()

file_name = 'Pepsi-logo.png'
image_folder = './images for logos detection/'
image_path = os.path.join(image_folder, file_name)

with io.open(image_path, 'rb') as image_file:
    content = image_file.read()

image = types.Image(content=content)
response = client.logo_detection(image=image)
logos = response.logo_annotations

for logo in logos:
    print('Logo Description:', logo.description)
    print('Confidence Score:', logo.score)
    print('-'*50)
    vertices = logo.bounding_poly.vertices
    print('Vertices Values {0}'.format(vertices))
    drawVertices(content, vertices, logo.description)