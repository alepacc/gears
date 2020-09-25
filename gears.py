import cv2
import numpy as np
from PIL import Image, ImageDraw

src_gear = "gears.png"
path = "img_result/"


# convert image by grayscale to binary
def imgToBin(img_array, height, width, threshold=100):
    result = np.zeros([height, width], dtype='uint8')
    for row in range(height):
        for col in range(width):
            if img_array[row, col] > threshold:
                result[row, col] = 1
            else:
                result[row, col] = 0
    return result

# Return an array that represent a circle
def get_circle(size):
    image = Image.new('L', (size, size))
    draw = ImageDraw.Draw(image)
    draw.ellipse([0, 0, size, size], fill='white')
    return np.array(image)


# Create ring with default thickness of 2 if = 0 return circle
def make_ring(n, thickness=2):
    small_ring = get_circle(n - thickness)
    big_ring = get_circle(n)
    res = np.array(big_ring)
    if thickness > 0:
        step = int(thickness / 2)
        for i in range(n - thickness):
            for j in range(n - thickness):
                res[i + step][j + step] = small_ring[i][j] ^ big_ring[i + step][j + step]
    return imgToBin(res, n, n)


# Convert np array to image and save it
def save_np_as_image(image, name):
    pil_image = Image.fromarray(np.uint8(image * 255), 'L')
    pil_image.save(path + name)


### MAIN ###

# convert source image RGB to gray scale
img_src = Image.open(src_gear).convert('L')
# convert image to array
img = np.array(img_src)
# convert  image to binary image
B = imgToBin(img, img_src.height, img_src.width)

# 1) erosion with hole ring
hole_size = 97
hole_ring = make_ring(hole_size, 2)
B1 = cv2.erode(B, hole_ring)
save_np_as_image(B1, 'B1.png')

# 2) dilatation with hole mask
hole_mask = make_ring(hole_size, 0)  # return circle of 97
B2 = cv2.dilate(B1, hole_mask)
save_np_as_image(B2, 'B2.png')

# 3) original image OR dilation
B3 = cv2.bitwise_or(B, B2)
save_np_as_image(B3, 'B3.png')

# 4) opening with gear_body
gear_size = 280
gear_body = make_ring(gear_size, 0)
erode = cv2.erode(B3, gear_body)  # open = erode --then--> dilate
B4 = cv2.dilate(erode, gear_body)
save_np_as_image(B4, 'B4_.png')


# 5) dilate B4 with sampling_ring_spacer --- circles dilated
ring_spacer_size = 10
sampling_ring_spacer = make_ring(ring_spacer_size, 0)
B5 = cv2.dilate(B4, sampling_ring_spacer)
save_np_as_image(B5, 'B5_.png')

# 6) circles dilated more
ring_width_size = 5
sampling_ring_width = make_ring(ring_width_size, 0)
B6 = cv2.dilate(B5, sampling_ring_width)
save_np_as_image(B6, 'B6_.png')


# 7) xor (B5, B6)
B7 = cv2.bitwise_xor(B5, B6)
save_np_as_image(B7, 'B7.png')

# 8) B and B7 --> ris i teeth
B8 = cv2.bitwise_and(B, B7)
save_np_as_image(B8, 'B8.png')

# 9) dilate with tip_spacing
tip_spacing_size = 15
tip_spacing = make_ring(tip_spacing_size, 0)
B9 = cv2.dilate(B8, tip_spacing)
save_np_as_image(B9, 'B9.png'.format(tip_spacing_size))

# 10)
B10 = cv2.subtract(B7, B9)
save_np_as_image(B10, 'B10_.png')
defect_cue_size = 35
defect_cue = make_ring(defect_cue_size, 0)
B10 = cv2.dilate(B10, defect_cue)
result = cv2.bitwise_or(B10, B9)
save_np_as_image(result, '_result.png')

