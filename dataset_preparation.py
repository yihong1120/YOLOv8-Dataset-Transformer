import shutil
import os
import random
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import cv2

def list_images(directory):
    """ List all images in a directory """
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(('.png', '.jpg', '.jpeg'))]

def calculate_label_positions(bg_width, bg_height, img, x, y):
    """ Calculate YOLOv8 label positions based on background dimensions """
    dw, dh = 1.0 / bg_width, 1.0 / bg_height
    # Calculate center, width and height
    cx, cy = (x + img.width / 2.0) * dw, (y + img.height / 2.0) * dh
    w, h = img.width * dw, img.height * dh
    return (cx, cy, w, h)

def apply_augmentations(image):
    """ Apply random image augmentations """
    if random.choice([True, False]):
        image = ImageOps.mirror(image)  # left-right flip
    if random.choice([True, False]):
        image = ImageOps.flip(image)  # top-bottom flip
    if random.choice([True, False]):
        image = image.rotate(random.randint(0, 360))  # rotation
    # add more augmentations as needed

    return image

def apply_more_augmentations(image):
    """ Apply additional image augmentations """
    if random.choice([True, False]):
        image = image.resize((random.randint(50, 150), random.randint(50, 150)))
    if random.choice([True, False]):
        image = image.rotate(random.randint(-30, 30), expand=True)
    if random.choice([True, False]):
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.5, 1.5))
    if random.choice([True, False]):
        image = ImageEnhance.Color(image).enhance(random.uniform(0.5, 1.5))
    # Add more augmentations as needed

    return image

def paste_images(background_image, images_to_paste, labels, output_dir, image_index):
    """ Paste images on a background image without overlap and save labels """
    background = Image.open(background_image).convert("RGBA")
    label_data = []
    pasted_areas = []  # List to keep track of pasted image areas

    for img_path, label in zip(images_to_paste, labels):
        img = Image.open(img_path).convert("RGBA")
        img = apply_augmentations(img)
        img = apply_more_augmentations(img)

        # Resize if necessary
        scale_factor = min(background.width / img.width, background.height / img.height)
        if scale_factor < 1:  # Only resize if image is larger than background
            new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
            img = img.resize(new_size, Image.ANTIALIAS)

        # Find a position to paste where the image does not overlap
        for _ in range(100):  # Try up to 100 times to find a non-overlapping position
            x = random.randint(0, background.width - img.width)
            y = random.randint(0, background.height - img.height)
            new_area = (x, y, x + img.width, y + img.height)

            if not any(overlaps(new_area, pasted_area) for pasted_area in pasted_areas):
                break
        else:
            continue  # Skip this image if no non-overlapping position is found

        background.alpha_composite(img, (x, y))
        pasted_areas.append(new_area)
        label_position = calculate_label_positions(background.width, background.height, img, x, y)
        label_data.append((label, *label_position))

    background = background.convert("RGB")
    final_image_path = os.path.join(output_dir, f'dataset_image_{image_index}.jpg')
    background.save(final_image_path)
    save_labels(output_dir, f'dataset_image_{image_index}.txt', label_data)
    return background

def overlaps(area1, area2):
    """ Check if two areas (x1, y1, x2, y2) overlap """
    return not (area1[2] <= area2[0] or area1[0] >= area2[2] or area1[3] <= area2[1] or area1[1] >= area2[3])


def save_labels(output_dir, label_file, label_data):
    """ Save the label data in YOLOv8 format """
    with open(os.path.join(output_dir, label_file), 'w') as file:
        for label in label_data:
            file.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")

def create_dataset(markers_dir, irrelevant_images_dir, output_dir, total_images, images_per_label=2):
    """ Create a dataset with specified number of images, each containing multiple labels """
    marker_folders = [os.path.join(markers_dir, f) for f in os.listdir(markers_dir) if os.path.isdir(os.path.join(markers_dir, f))]
    all_marker_images = {folder: list_images(folder) for folder in marker_folders}
    all_irrelevant_images = list_images(irrelevant_images_dir)
    random.shuffle(all_irrelevant_images)  # Shuffle the irrelevant images

    for image_index in range(total_images):
        # Select an unused irrelevant image as background
        background_image = all_irrelevant_images.pop() if all_irrelevant_images else random.choice(list_images(irrelevant_images_dir))

        # Collect images from different labels
        selected_images = []
        labels = []

        # Randomly select a random number (4 to 10) of folders
        num_folders_to_select = random.randint(4, 10)
        random.shuffle(marker_folders)
        for folder in marker_folders[:num_folders_to_select]:
            folder_images = all_marker_images[folder]
            if folder_images:  # Check if there are unused images in the folder
                selected_from_folder = [folder_images.pop() for _ in range(min(len(folder_images), images_per_label))]
            else:  # If all images in the folder have been used, randomly select
                selected_from_folder = random.sample(list_images(folder), images_per_label)

            selected_images.extend(selected_from_folder)
            labels.extend([os.path.basename(folder)] * len(selected_from_folder))

        final_image = paste_images(background_image, selected_images, labels, output_dir, image_index)


def split_dataset(output_dir, train_ratio=0.8):
    images = [f for f in os.listdir(output_dir) if f.endswith('.jpg')]
    random.shuffle(images)  # Shuffle images before splitting

    num_train = int(len(images) * train_ratio)
    train_images = images[:num_train]
    val_images = images[num_train:]

    # Create directories for train and val sets
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # Function to move images and labels
    def move_files(files, destination):
        for f in files:
            # Move image
            shutil.move(os.path.join(output_dir, f), os.path.join(destination, f))
            # Move corresponding label file
            label_file = f.replace('.jpg', '.txt')
            shutil.move(os.path.join(output_dir, label_file), os.path.join(destination, label_file))

    # Move files to respective directories
    move_files(train_images, train_dir)
    move_files(val_images, val_dir)

    # Create file lists for YOLOv8 training
    with open(os.path.join(output_dir, 'train.txt'), 'w') as file:
        file.writelines([os.path.join(train_dir, f) + '\n' for f in train_images])

    with open(os.path.join(output_dir, 'val.txt'), 'w') as file:
        file.writelines([os.path.join(val_dir, f) + '\n' for f in val_images])


# Parameters
markers_main_path = 'train20X20'
irrelevant_images_path = 'irrelevant'
output_path = 'output'
total_dataset_images = 1000  # specify the desired number of images in the dataset

# After dataset creation
create_dataset(markers_main_path, irrelevant_images_path, output_path, total_dataset_images)

# Split the dataset
split_dataset(output_path)