from PIL import Image
import dlib
import os
import numpy as np

target_size = (48, 48)

input_dir = "engaged"
output_dir = "engaged_preprocessed"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

face_detector = dlib.get_frontal_face_detector()
image_counter = 1

for filename in os.listdir(input_dir):
    if filename.endswith((".jpg", ".jpeg", ".png")):
        image = Image.open(os.path.join(input_dir, filename))
        grayscale_image = image.convert("L")
        image_array = np.array(grayscale_image)
        faces = face_detector(image_array)

        if len(faces) > 0:
            face = faces[0]
            x, y, w, h = face.left(), face.top(), face.width(), face.height()
            cropped_resized_image = Image.fromarray(image_array[y:y+h, x:x+w]).resize(target_size)
            output_filename = f"{image_counter}.png"
            cropped_resized_image.save(os.path.join(output_dir, output_filename))
            print(f"Processed: {filename} => Saved as {output_filename}")
            image_counter += 1
