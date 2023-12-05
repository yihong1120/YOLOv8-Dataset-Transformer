import argparse
import shutil
import os
import random
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import cv2
import yaml
from tqdm import tqdm


class DatasetCreator:
    def __init__(
        self,
        markers_dir,
        irrelevant_images_dir,
        output_dir,
        total_images,
        # images_per_label=2,
    ):
        self.markers_dir = markers_dir
        self.irrelevant_images_dir = irrelevant_images_dir
        self.output_dir = output_dir
        self.total_images = total_images
        # self.images_per_label = images_per_label

        self.marker_folders = [
            os.path.join(markers_dir, f)
            for f in os.listdir(markers_dir)
            if os.path.isdir(os.path.join(markers_dir, f))
        ]
        self.all_marker_images = {
            folder: self.list_images(folder) for folder in self.marker_folders
        }
        self.all_irrelevant_images = self.list_images(irrelevant_images_dir)
        random.shuffle(self.all_irrelevant_images)  # Shuffle the irrelevant images

    @staticmethod
    def list_images(directory):
        """List all images in a directory"""
        return [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ]

    @staticmethod
    def calculate_label_positions(bg_width, bg_height, img, x, y):
        """Calculate YOLOv8 label positions based on background dimensions"""
        dw, dh = 1.0 / bg_width, 1.0 / bg_height
        # Calculate center, width and height
        cx, cy = (x + img.width / 2.0) * dw, (y + img.height / 2.0) * dh
        w, h = img.width * dw, img.height * dh
        return (cx, cy, w, h)

    @staticmethod
    def apply_augmentations(image):
        """Apply random image augmentations"""
        if random.choice([True, False]):
            image = ImageOps.mirror(image)  # left-right flip
        if random.choice([True, False]):
            image = ImageOps.flip(image)  # top-bottom flip
        if random.choice([True, False]):
            image = image.rotate(random.randint(0, 360))  # rotation
        # add more augmentations as needed

        return image

    @staticmethod
    def apply_more_augmentations(image):
        """Apply additional image augmentations"""
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

    def paste_images(self, background_image, images_to_paste, labels, image_index):
        """Paste images on a background image without overlap and save labels"""

        # 这段代码检查目标文件夹是否存在，如果不存在则创建该文件夹
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        background = Image.open(background_image).convert("RGBA")
        label_data = []
        pasted_areas = []  # List to keep track of pasted image areas

        for img_path, label in zip(images_to_paste, labels):
            img = Image.open(img_path).convert("RGBA")
            img = self.apply_augmentations(img)
            img = self.apply_more_augmentations(img)

            # Resize if necessary
            scale_factor = min(
                background.width / img.width, background.height / img.height
            )
            if scale_factor < 1:  # Only resize if image is larger than background
                new_size = (
                    int(img.width * scale_factor),
                    int(img.height * scale_factor),
                )
                img = img.resize(new_size, Image.Resampling.LANCZOS) 

            # Find a position to paste where the image does not overlap
            for _ in range(
                100
            ):  # Try up to 100 times to find a non-overlapping position
                x = random.randint(0, background.width - img.width)
                y = random.randint(0, background.height - img.height)
                new_area = (x, y, x + img.width, y + img.height)

                if not any(
                    self.overlaps(new_area, pasted_area) for pasted_area in pasted_areas
                ):
                    break
            else:
                continue  # Skip this image if no non-overlapping position is found

            background.alpha_composite(img, (x, y))
            pasted_areas.append(new_area)
            label_position = self.calculate_label_positions(
                background.width, background.height, img, x, y
            )
            label_data.append((label, *label_position))

        background = background.convert("RGB")
        final_image_path = os.path.join(
            self.output_dir, f"dataset_image_{image_index}.jpg"
        )
        background.save(final_image_path)
        self.save_labels(f"dataset_image_{image_index}.txt", label_data)
        return background

    @staticmethod
    def overlaps(area1, area2):
        """Check if two areas (x1, y1, x2, y2) overlap"""
        return not (
            area1[2] <= area2[0]
            or area1[0] >= area2[2]
            or area1[3] <= area2[1]
            or area1[1] >= area2[3]
        )

    def save_labels(self, label_file, label_data):
        """Save the label data in YOLOv8 format"""
        with open(os.path.join(self.output_dir, label_file), "w") as file:
            for label in label_data:
                # Convert the label name to its corresponding index
                label_index = self.label_index.get(label[0], -1)
                if label_index == -1:
                    continue  # Skip if label name not found in index

                file.write(
                    f"{label_index} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n"
                )

    def create_dataset(self):
        """Create a dataset with specified number of images, each containing multiple labels"""
        for image_index in tqdm(range(self.total_images), desc="Creating dataset:"):
            # Select an unused irrelevant image as background
            background_image = (
                self.all_irrelevant_images.pop()
                if self.all_irrelevant_images
                else random.choice(self.list_images(self.irrelevant_images_dir))
            )

            # Collect images from different labels
            selected_images = []
            labels = []

            # Randomly select a random number (4 to 10) of folders
            num_folders_to_select = random.randint(4, 10)
            random.shuffle(self.marker_folders)
            for folder in self.marker_folders[:num_folders_to_select]:
                folder_images = self.all_marker_images[folder]
                if folder_images:  # Check if there are unused images in the folder
                    # Randomly decide the number of images to select from each folder (between 2 to 4)
                    num_images = random.randint(2, 4)
                    selected_from_folder = [
                        folder_images.pop()
                        for _ in range(min(len(folder_images), num_images))
                    ]
                else:  # If all images in the folder have been used, randomly select
                    selected_from_folder = random.sample(
                        self.list_images(folder), num_images
                    )

                selected_images.extend(selected_from_folder)
                labels.extend([os.path.basename(folder)] * len(selected_from_folder))

            self.paste_images(background_image, selected_images, labels, image_index)

    def split_dataset(self, train_ratio=0.8):
        images = [f for f in os.listdir(self.output_dir) if f.endswith(".jpg")]
        random.shuffle(images)  # Shuffle images before splitting

        num_train = int(len(images) * train_ratio)
        train_images = images[:num_train]
        val_images = images[num_train:]

        # Create directories for train and val sets
        train_dir = os.path.join(self.output_dir, "train")
        val_dir = os.path.join(self.output_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        # Function to move images and labels
        def move_files(files, destination):
            for f in files:
                # Move image
                shutil.move(
                    os.path.join(self.output_dir, f), os.path.join(destination, f)
                )
                # Move corresponding label file
                label_file = f.replace(".jpg", ".txt")
                shutil.move(
                    os.path.join(self.output_dir, label_file),
                    os.path.join(destination, label_file),
                )

        # Move files to respective directories
        move_files(tqdm(train_images, desc="Moving train images:"), train_dir)
        move_files(tqdm(val_images, desc="Moving validation images:"), val_dir)

        # Create file lists for YOLOv8 training
        with open(os.path.join(self.output_dir, "train.txt"), "w") as file:
            file.writelines([os.path.join(train_dir, f) + "\n" for f in train_images])

        with open(os.path.join(self.output_dir, "val.txt"), "w") as file:
            file.writelines([os.path.join(val_dir, f) + "\n" for f in val_images])

    def create_label_index(self):
        """ Create a mapping from label names to indices. """
        self.label_index = {name: idx for idx, name in enumerate(os.listdir(self.markers_dir))}

    def create_data_yaml(self, train_file="train.txt", val_file="val.txt"):
        # 獲取當前工作目錄的絕對路徑
        current_path = os.path.abspath(os.path.join(self.output_dir, ".."))

        # 計算類別數量，假設有一個額外的類別
        n_classes = len(self.marker_folders) + 1

        # 創建名稱與索引的字典
        names_dict = {idx: name for idx, name in enumerate(os.listdir(self.markers_dir))}

        data = dict(
            path=current_path,
            train=os.path.join(self.output_dir, train_file),
            val=os.path.join(self.output_dir, val_file),
            nc=n_classes,
            names=names_dict
        )

        with open(
            os.path.join(self.output_dir, "data.yaml"), "w", encoding="utf8"
        ) as file:
            yaml.dump(data, file, default_flow_style=False, sort_keys=False)


if __name__ == "__main__":
    # Create ArgumentParser object
    parser = argparse.ArgumentParser(
        description="Create dataset and split into training and validation sets."
    )

    # Add arguments
    parser.add_argument(
        "--markers",
        type=str,
        required=True,
        help="Path to directory containing markers.",
    )
    parser.add_argument(
        "--irrelevant",
        type=str,
        required=True,
        help="Path to directory containing irrelevant images.",
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Path to save the output dataset."
    )
    parser.add_argument(
        "--total_images",
        type=int,
        default=1000,
        help="The total desired number of images in the dataset.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.8,
        help="Ratio of train data to total data.",
    )

    # Parse arguments from command-line
    args = parser.parse_args()

    # Use the command line arguments in the program
    creator = DatasetCreator(
        args.markers, args.irrelevant, args.output, args.total_images
    )
    
    creator.create_label_index()  # Initialise label index mapping
    creator.create_dataset()
    creator.split_dataset(train_ratio=args.train_ratio)
    creator.create_data_yaml()  

    # python dataset_preparation.py --markers train20X20 --irrelevant irrelevant --output output --total_images 7000 --train_ratio 0.9