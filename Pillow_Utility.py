from PIL import Image, ImageDraw

def draw_borders(pillow_image, bounding, color, caption='', confidence_score=0, border_width=3):
    draw = ImageDraw.Draw(pillow_image)
    draw.polygon([
        (bounding.normalized_vertices[0].x * pillow_image.width, bounding.normalized_vertices[0].y * pillow_image.height),
        (bounding.normalized_vertices[1].x * pillow_image.width, bounding.normalized_vertices[1].y * pillow_image.height),
        (bounding.normalized_vertices[2].x * pillow_image.width, bounding.normalized_vertices[2].y * pillow_image.height),
        (bounding.normalized_vertices[3].x * pillow_image.width, bounding.normalized_vertices[3].y * pillow_image.height)],
        outline=color, width=border_width)

    draw.text((bounding.normalized_vertices[0].x * pillow_image.width,
               bounding.normalized_vertices[0].y * pillow_image.height), caption, fill=color)

    # insert confidence score
    draw.text((bounding.normalized_vertices[0].x * pillow_image.width, bounding.normalized_vertices[0].y * pillow_image.height + 20),
              'Confidence Score: {0:.2f}%'.format(confidence_score), fill=color)
    return pillow_image
