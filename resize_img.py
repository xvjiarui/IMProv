from PIL import Image
import os

def convert_to_jpg(root_dir, target_size=(224, 224)):
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpeg', '.gif', '.bmp', '.tiff')):
                try:
                    with Image.open(os.path.join(root, file)) as img:
                        # Convert to RGB if the image is not already in this mode
                        if img.mode != 'RGB':
                            img = img.convert('RGB')

                        # Resize the image
                        img = img.resize(target_size)

                        # Save the image in JPG format
                        new_filename = os.path.splitext(file)[0] + '.jpg'
                        img.save(os.path.join(root, new_filename))

                        print(f"Converted and resized {file} to {new_filename}")
                except Exception as e:
                    print(f"Failed to convert {file}: {e}")

# Example usage:
convert_to_jpg('figs', (384, 384))

# Note: This script assumes that the PIL library is installed. 
# You can install it via pip: pip install pillow
# Replace '/path/to/your/folder' with the actual path to the folder containing the images you want to convert.
