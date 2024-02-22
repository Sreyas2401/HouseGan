import PIL
from PIL import Image
import PIL.ImageOps  

image = Image.open('../shapes/raw_images/hand_drawn.png')
if image.mode == 'RGBA':
    r,g,b,a = image.split()
    rgb_image = Image.merge('RGB', (r,g,b))

    inverted_image = PIL.ImageOps.invert(rgb_image)

    r2,g2,b2 = inverted_image.split()

    final_transparent_image = Image.merge('RGBA', (r2,g2,b2,a))

    final_transparent_image.save('../shapes/raw_images/hand_drawn.png')

else:
    inverted_image = PIL.ImageOps.invert(image)
    inverted_image.save('../shapes/raw_images/hand_drawn.png')