import os

def list_images_in_current_folder(output_file="image_list.txt"):
    # Get the folder where this script is located
    script_folder = os.path.dirname(os.path.abspath(__file__))
    
    # Define common image file extensions
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}
    
    try:
        # Get list of all image files in the script's directory
        image_files = [f for f in os.listdir(script_folder) 
                       if os.path.isfile(os.path.join(script_folder, f)) and f.lower().endswith(tuple(image_extensions))]

        # Write image filenames to the output file in the same directory
        output_path = os.path.join(script_folder, output_file)
        with open(output_path, "w") as file:
            for image in image_files:
                file.write(image + "\n")

        print(f"Image list saved to {output_path}. Found {len(image_files)} images.")

    except Exception as e:
        print(f"Error: {e}")

# Run the function
list_images_in_current_folder()