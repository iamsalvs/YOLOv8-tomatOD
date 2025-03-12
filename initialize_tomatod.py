import random
import os

def tomatodInitialize():
    """
    Initialize the tomatOD dataset by creating text files that list image paths
    for the train, validation, and test sets.

    Assumes the following directory structure:
    - tomatOD_yolo/
        - data.yaml
        - train/
            - images/
            - labels/
        - val/
            - images/
            - labels/
        - test/
            - images/
            - labels/

    This function creates:
    - tomatOD_yolo/tomatOD_train.txt
    - tomatOD_yolo/tomatOD_val.txt
    - tomatOD_yolo/tomatOD_test.txt
    """
    import os

    base_dir = 'tomatOD_yolo'
    sets = ['train', 'val', 'test']
    
    for s in sets:
        images_dir = os.path.join(base_dir, s, 'images')
        txt_file = os.path.join(base_dir, f'tomatOD_{s}.txt')
        
        # List image files with common extensions
        images = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        images.sort()  # Optional: sort the list if desired
        
        with open(txt_file, 'w') as f:
            for img in images:
                # Save the relative path with respect to base_dir
                image_path = os.path.join(s, 'images', img)
                f.write(image_path + '\n')
