from PIL import Image

def rescale_image(input_image_path, output_image_path, size=(224, 224)):
    # Open an image file
    with Image.open(input_image_path) as img:
        # Rescale the image
        img_rescaled = img.resize(size)
        # Save the rescaled image
        img_rescaled.save(output_image_path)
        print(f"Image has been rescaled to {size} and saved as {output_image_path}")

# Example usage
input_image_path = './gradcam_overlay_front.png'  # Change this to your image path
output_image_path = './gradcam_overlay_front_224.png'  # Change this to where you want to save the new image
rescale_image(input_image_path, output_image_path)
input_image_path = './gradcam_overlay_lat.png'  # Change this to your image path
output_image_path = './gradcam_overlay_lat_224.png'  # Change this to where you want to save the new image
rescale_image(input_image_path, output_image_path)