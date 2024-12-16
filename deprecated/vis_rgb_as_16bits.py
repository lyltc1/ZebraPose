import cv2
import sys
sys.path.append('/home/ZebraPose/zebrapose')

from binary_code_helper.class_id_encoder_decoder import RGB_image_to_class_id_image, class_id_image_to_class_code_images, class_code_images

img_path = "/home/data/vis/img.jpg"
img = cv2.imread(img_path)
class_id_image = RGB_image_to_class_id_image(img)
class_code_images = class_id_image_to_class_code_images(class_id_image, iteration=8, number_of_class=256)

print(class_code_images.shape)