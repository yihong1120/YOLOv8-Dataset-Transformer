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
    python dataset_preparation.py --markers train20X20 --irrelevant irrelevant --output output --total_images 1000 --train_ratio 0.8
    ```

    ![thumbnail.jpg](images/thumbnail.jpg)
    
    New images shall be generated by the script, you can refer to the image above:

3. **Train Your Model**:

    ```bash
    python train.py --data_config path/to/data.yaml --epochs 100 --model_name yolov8n.pt
    ```

4. **Evaluate and Export Your Model**:

    Validate, predict, and export your model using options in the `train.py` script.

## To do

Train and demonstrate the model and computed the parameters of experiments.

## Contributing

Contributions to the YOLOv8-Dataset-Transformer are welcome! Please read our [Contributing Guidelines](CONTRIBUTING.md) for more information.
We fetch the [train20X20 dataset](https://github.com/apoorva-dave/LicensePlateDetector/tree/master/train20X20) from [apoorva-dave](https://github.com/apoorva-dave) and [irrelevant](./irrelevant) images from Google image.

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.

### Acknowledgements

- Thanks to the [Ultralytics team](https://ultralytics.com) for the YOLOv8 model.
- Special thanks to all contributors and maintainers of this project.