import io
from PIL import Image, ImageDraw, ImageFont

def drawVertices(image_source, vertices, display_text=''):
    pillow_img = Image.open(io.BytesIO(image_source))

    draw = ImageDraw.Draw(pillow_img)
    for i in range(len(vertices) - 1):
        draw.line(((vertices[i].x, vertices[i].y), (vertices[i + 1].x, vertices[i + 1].y)),
                fill='green',
                width=8
        )

    draw.line(((vertices[len(vertices) - 1].x, vertices[len(vertices) - 1].y),
               (vertices[0].x, vertices[0].y)),
               fill='green',
               width=8
    )

    # Use a built-in font instead of loading a font file
    font = ImageFont.load_default()
    draw.text((vertices[0].x + 10, vertices[0].y),
              font=font, text=display_text, 
              fill=(255, 255, 255))
    pillow_img.show()
    
