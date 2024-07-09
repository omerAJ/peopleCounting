# People Counting

## Project Aim
The aim of this project is to develop a robust system for counting people in crowded environments using advanced computer vision techniques. This involves using the YOLOv8 model, which is known for its high performance in object detection tasks, to detect and track individuals in video feeds. The project focuses on accurately counting the number of people crossing a predefined line, making it applicable for scenarios such as traffic monitoring and managing crowd flow in public areas.


## Project Overview
The primary goal of this repository is to implement a people counting system using the YOLOv8 object detection model from Ultralytics. The model is trained on the Crowd Human dataset [1], the CrowdHuman dataset is large, rich-annotated and contains high diversity, ensuring it can generalize well across different environments with varied crowd densities. The system processes video feeds, detects and tracks people, and counts how many cross a designated line in the video.


## Table of Contents
- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
- [Usage](#usage)
  - [Detection](#detection)
  - [Training](#training)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Getting Started

### Prerequisites
- Python
- YOLOv8 (Ultralytics)

### Installation
1. Clone the repository:
git clone https://github.com/omerAJ/peopleCounting.git
cd people-counting

2. Install required Python libraries:
pip install -r requirements.txt


## Usage

### Detection

- **Just detect and count**
  - Use the provided script in Scripts/counting.py to run the YOLOv8 model on your video feed. This script loads the pre-trained YOLOv8 model and applies it to detect and count people in the video, it tracks the detected individuals across frames and counts how many cross a predefined line in the video feed. Adjust the line coordinates in the script based on your specific requirements.

- **Count and collect GT**
  - In addition to detect and count this code can also be used to collect GT counts to evaluate performance of model. As you watch the video, the model counts are automatically incremented and you can use key presses to count the actual people in/out.

### Training

#### Dataset

The crowd human dataset used for training the YOLO is placed in D://omer/crowdHuman
The folder includes:
1. original data
2. data converted to YOLO format
3. annotation of head of human

## Contributing
Full credits to the two mentioned papers.

[1] @article{shao2018crowdhuman,
    title={CrowdHuman: A Benchmark for Detecting Human in a Crowd},
    author={Shao, Shuai and Zhao, Zijian and Li, Boxun and Xiao, Tete and Yu, Gang and Zhang, Xiangyu and Sun, Jian},
    journal={arXiv preprint arXiv:1805.00123},
    year={2018}
  }

## Acknowledgements
- This project was done at Centre for Urban Informatics, Technology, and Policy (CITY @ LUMS)

