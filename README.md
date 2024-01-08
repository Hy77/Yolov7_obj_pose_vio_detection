# README for Violence Behavior Detection Project based on YOLOv7

## Introduction

This repository hosts a custom implementation for detecting violence behavior, integrating pose detection with YOLOv7's robust object detection capabilities. It's designed to identify and analyze violent behaviors in various settings using advanced machine learning techniques.

## Project Components

### Python Scripts
The repository includes several Python scripts, each serving a unique purpose in the violence detection workflow:

- **`check_GPU.py`**: Automatically detects the presence of a GPU and configures the system to utilize it for enhanced performance. In the absence of a GPU, the script falls back to using the CPU.
- **`pose_detect.py`**: Implements basic pose detection leveraging the YOLOv7 model. It serves as a foundational script for understanding pose detection mechanics.
- **`obj_pose_detect.py`**: This script merges YOLOv7's object detection capabilities with pose detection, enabling simultaneous detection of objects and human poses.
- **`obj_pose_vio_detect.py`**: Extends `obj_pose_detect.py` by incorporating an external violence detection model from Roboflow. Note that this script is subject to request limitations due to its reliance on an external model.
- **`obj_pose_vio_detect_local.py`**: A self-contained version of the violence detection script using a locally trained model. It eliminates the dependency on external APIs, thus removing any request limitations and ensuring efficient processing.

### Dependencies

- **YOLOv7**: The backbone of the project, providing advanced object detection capabilities.
- **Roboflow**: Used for obtaining and processing the violence detection dataset.

## Setup Instructions

### Clone YOLOv7 Repository
Start by cloning the YOLOv7 repository to get the necessary object detection framework:
```bash
git clone https://github.com/WongKinYiu/yolov7.git
```

### Download Dataset
The violence detection model is trained on a specific dataset available through Roboflow. You can download it using one of the following methods:

**Method 1: Direct Download**
```bash
curl -L "https://universe.roboflow.com/ds/P2nFEF04Vm?key=Vj4qEKv7pf" > roboflow.zip; unzip roboflow.zip; rm roboflow.zip
```

**Method 2: Using Roboflow Python Package**
```python
!pip install roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("east-west-uniersity").project("violance-nonviolance")
dataset = project.version(5).download("yolov7")
```

### Download Pretrained Models
The project uses various pretrained models. Download them using the following links:

- YOLOv7 Pretrained Model: [yolov7.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt)
- YOLOv7 Pose Model: [yolov7-w6-pose.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7-w6-pose.pt)
- For additional training: [yolov7_training.pt](https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7_training.pt)

Certainly, I can add specific examples to the "Usage" section of the README to illustrate how to run the scripts. Here's how it can be presented:

## Usage

### Environment Setup
Follow the setup instructions to prepare your environment.

### Run Scripts
Execute the desired Python script based on your specific requirements. Here are some examples to guide you:

1. **Check GPU Utilization**
   To check and utilize your system's GPU, run:
   ```
   python check_GPU.py
   ```
   This script will automatically detect if a GPU is available and use it. If not, it will default to using the CPU.

2. **Basic Pose Detection**
   For running basic pose detection:
   ```
   python pose_detect.py --source your_image_or_video_path
   ```
   Replace `your_image_or_video_path` with the path to the image or video file you want to analyze.

3. **Object and Pose Detection**
   To perform both object and pose detection:
   ```
   python obj_pose_detect.py --source your_image_or_video_path
   ```
   Ensure to specify the path to your input file.

4. **Violence Detection**
   You have two options for violence detection:

   - **Using External Model**:
     To use the external Roboflow model for violence detection, run:
     ```
     python obj_pose_vio_detect.py --source your_image_or_video_path
     ```
   - **Using Local Model**:
     If you prefer to use the locally trained model, execute:
     ```
     python obj_pose_vio_detect_local.py --source inference/images/xxx.jpg --device 0/cpu
     ```
     Here, `inference/images/xxx.jpg` should be replaced with your input file path, and `0/cpu` should be replaced with `0` to use the GPU or `cpu` to use the CPU.

Each script comes with various command-line arguments to customize your run, such as `--source` for specifying the input file, and `--device` to choose between GPU and CPU processing. Ensure to replace the placeholder paths with actual paths to your images, videos, or datasets.
## Additional Resources

- **Training Documentation**: For an in-depth understanding of the model's training process, refer to the [Training Documentation](https://github.com/Hy77/Yolov7_obj_pose_vio_detection/blob/main/Training_Documentation.md).
- **YOLOv7 Official Repository**: Essential for understanding the underlying object detection framework. Available at [https://github.com/WongKinYiu/yolov7.git](https://github.com/WongKinYiu/yolov7.git).
- **Violence-Nonviolence Dataset**: The dataset used for training the model, accessible at [https://universe.roboflow.com/east-west-uniersity/violance-nonviolance](https://universe.roboflow.com/east-west-uniersity/violance-nonviolance).

## Note

This repository does not replicate all components of the YOLOv7 official repository. Ensure to follow the setup instructions for a complete and functional setup, especially if you're interested in further training or custom implementations.
