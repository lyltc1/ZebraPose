import cv2
import sys
sys.path.append('/home/ZebraPose/zebrapose')

from binary_code_helper.class_id_encoder_decoder import RGB_image_to_class_id_image, class_id_image_to_class_code_images

img_path = "/home/data/vis/img.jpg"
img = cv2.imread(img_path)
class_id_image = RGB_image_to_class_id_image(img)
class_code_images = class_id_image_to_class_code_images(class_id_image, iteration=8, number_of_class=256)

# shape of class_code_images is (H, W, 8)
# save each channel as a separate image
for i in range(class_code_images.shape[2]):
    channel_image = class_code_images[:, :, i] * 255.
    output_path = f"/home/data/vis/class_code_channel_{i}.png"
    cv2.imwrite(output_path, channel_image)
    print(f"Saved {output_path}")