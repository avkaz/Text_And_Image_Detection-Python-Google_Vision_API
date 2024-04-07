import os
import io
from google.cloud import vision
from google.cloud.vision_v1 import types

# Set the path to your service account credentials JSON file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = r'Service_account_token.json'

# Initialize the Vision API client
client = vision.ImageAnnotatorClient()

##################### detect handwritten text on images #####################

def detectTextLocalImage(img_folder):
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
                # Perform text detection on the image
                response = client.text_detection(image=image)
                
                # Extract text annotations
                texts = response.full_text_annotation.text
                detected_text = texts if texts else "No text detected"
                
                # Write the detected text to a text file in the results folder
                output_filename = os.path.splitext(file_name)[0] + ".txt"
                output_path = os.path.join(result_folder, output_filename)
                with open(output_path, 'w') as text_file:
                    text_file.write(detected_text)
                
                # Check for any errors in the response
                if response.error.message:
                    print(f"Error for {file_name}: {response.error.message}")
            except Exception as e:
                print(f"An error occurred for {file_name}: {e}")

# Define the path to the folder containing images
img_folder = os.path.abspath("images with handwritten text")

# Call the detectTextLocalImage function with the image folder path
detectTextLocalImage(img_folder)

##################### detect handwritten text with confidence #####################

def detectTextLocalImageWithConfidence(img_path):
    # Read the image file
    with io.open(img_path, 'rb') as image_file:
        content = image_file.read()

    # Create an Image object
    image = types.Image(content=content)

    try:
        # Perform text detection on the image
        response = client.document_text_detection(image=image)
        pages = response.full_text_annotation.pages

        # Create a results_with_confidence folder inside the image folder if it doesn't exist
        result_folder = os.path.join(os.path.dirname(img_path), 'results_with_confidence')
        os.makedirs(result_folder, exist_ok=True)

        for page in pages:
            for block in page.blocks:
                # For each block confidence
                block_confidence = block.confidence
                for paragraph in block.paragraphs:
                    # For each paragraph confidence
                    paragraph_confidence = paragraph.confidence
                    
                    for word in paragraph.words:
                        word_text = ''.join([symbol.text for symbol in word.symbols])
                        word_confidence = word.confidence

                        for symbol in word.symbols:
                            symbol_text = symbol.text
                            symbol_confidence = symbol.confidence
                            
                # Save results to a file
                output_filename = os.path.splitext(os.path.basename(img_path))[0] + "_result.txt"
                output_path = os.path.join(result_folder, output_filename)
                with open(output_path, 'a') as text_file:
                    text_file.write(f"Block confidence: {block_confidence}\n")
                    text_file.write(f"Paragraph confidence: {paragraph_confidence}\n")
                    text_file.write(f"Word text: {word_text} (confidence: {word_confidence})\n")
                    text_file.write(f"\tSymbol: {symbol_text} (confidence: {symbol_confidence})\n")
    except Exception as e:
        print(f"An error occurred for {img_path}: {e}")

# Define the path to the image file
img_path = os.path.abspath("images with handwritten text/micah-boswell-00nHr1Lpq6w-unsplash.jpg")

# Call the detectTextLocalImageWithConfidence function with the image file path
detectTextLocalImageWithConfidence(img_path)
