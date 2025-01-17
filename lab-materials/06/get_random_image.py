import os
import random

def get_random_image(path: str = "06-wildfire-sample") -> str:
    """
    Search through all subfolders of the specified path for image files,
    pick one at random, and return its absolute path.

    Args:
        path (str): The root directory to search for image files. Defaults to a specific path.

    Returns:
        str: The absolute path of a randomly selected image file.
    """
    # Define valid image file extensions
    valid_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff"}

    # List to hold all image file paths
    image_files = []

    # Walk through the directory and collect image files
    for root, _, files in os.walk(path):
        if "__MACOSX" in root:
            continue
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                image_files.append(os.path.join(root, file))

    # Check if any image files were found
    if not image_files:
        raise FileNotFoundError("No image files found in the specified directory and its subfolders.")

    # Select and return a random image file
    return random.choice(image_files)