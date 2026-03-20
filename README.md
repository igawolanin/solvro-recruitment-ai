# Handwritten and Digitally Generated Image Classification

### Overview

This project explores image classification using transfer learning. It compares two approaches: fine-tuning the entire 
model or freezing the backbone and fine-tuning only the last layer to evaluate which performs better. Project uses a pretrained ResNet18 model.

### Dataset

The dataset used in this project comes from Kaggle:

[Kaggle dataset – Simple hand-drawn and digitized images](https://www.kaggle.com/datasets/gergvincze/simple-hand-drawn-and-digitized-images/data)

It consists of grayscale images representing different classes of symbols. The images are organized into class-specific directories and vary in visual style.

### Project Structure

The repository is divided into separate files, each responsible for a different part of the pipeline. Jupyter notebook is used for experiments and analysis:
- `data/`: Contains subdirectories named after class label, each containing files of jpg format with images.
- `src/`: Contains the source code, organized into files:
    - `research.ipynb`: Executes and analyses experiments.
    - `config.py`: Contains configuration.
    - `cam.py`: Computations for Class Activation Mapping.
    - `data.py`: Loads, splits and transforms data.
    - `eval.py`: Performs evaluation. 
    - `models.py`: Defines models.
    - `train.py`: Handles training.
    - `visualization.py`: Visualizes dataset attributes and results.

### Training Approaches

Two training strategies were compared:

- **Full fine-tuning** – all model layers are trained  
- **Frozen backbone** – only the final classification layer is trained while the feature extractor remains fixed

### License

This project is released under the GNU GPL v3 license. More information can be found in the `LICENSE.txt` file.

### Contact

If you have any questions or feedback, feel free to reach out via GitHub.