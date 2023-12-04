# YOLOv8-Dataset-Transformer

YOLOv8-Dataset-Transformer is an integrated solution for transforming image classification datasets into object detection datasets, followed by training with the state-of-the-art YOLOv8 model. This toolkit simplifies the process of dataset augmentation, preparation, and model training, offering a streamlined path for custom object detection projects.

## Features

- **Dataset Conversion**: Converts standard image classification datasets into YOLOv8 compatible object detection datasets.
- **Image Augmentation**: Applies a variety of augmentations to enrich the dataset, improving model robustness.
- **Model Training and Validation**: Facilitates the training and validation of YOLOv8 models with custom datasets.
- **Model Exporting**: Supports exporting trained models to different formats like ONNX for easy deployment.

## Getting Started

### Prerequisites

- Python 3.8 or later
- PyTorch 1.8 or later
- YOLOv8 dependencies (refer to [YOLOv8 documentation](https://github.com/ultralytics/yolov8))

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/[YourUsername]/YOLOv8-Dataset-Transformer.git
cd YOLOv8-Dataset-Transformer
```

Install the required packages:

```bash
pip install -r requirements.txt
```

### Usage

1. **Prepare Your Dataset**: Place your image classification dataset in the designated folders.

2. **Run the Dataset Preparation Script**:

    ```bash
    python dataset_preparation.py --markers_dir [path_to_markers] --irrelevant_images_dir [path_to_irrelevant_images] --output_dir [path_to_output] --total_images [number_of_images]
    ```

3. **Train Your Model**:

    ```bash
    python train.py --data_config path/to/data.yaml --epochs 100 --model_name yolov8n.pt
    ```

4. **Evaluate and Export Your Model**:

    Validate, predict, and export your model using options in the `train.py` script.

### Contributing

Contributions to the YOLOv8-Dataset-Transformer are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for more information.

### License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Acknowledgements

- Thanks to the [Ultralytics team](https://ultralytics.com) for the YOLOv8 model.
- Special thanks to all contributors and maintainers of this project.