from PIL import Image
import os
input_folder = "/home/nvidia/finalProject/locks/KeyL/train"
output_folder = "train_kL_output4"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
counter = 1

for filename in os.listdir(input_folder):
    if any(filename.lower().endswith(ext) for ext in image_extensions):
        try:
            input_path = os.path.join(input_folder, filename)
            img = Image.open(input_path)

            resized_img = img.resize((512,512))

            _, ext = os.path.splitext(filename)

            new_filename = f"{counter:03d}{ext}"

            output_path = os.path.join(output_folder, new_filename)
            resized_img.save(output_path)

            print(f"checkmark Resized: {filename} -> {new_filename}")

            counter += 1 

        except Exception as e:
            print(f"X Error with {filename}: {e}")


print("\nAll done. check the resized+images folder for pics")


