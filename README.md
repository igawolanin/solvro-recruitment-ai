# Handwritten and Digitally Generated Image Classification

### Overview

This project explores image classification using transfer learning. It compares two approaches: fine-tuning the entire 
model or freezing the backbone and fine-tuning only the last layer to evaluate which performs better. The dataset consists of grayscale images.
### Project Structure

The repository is divided into separate files, each responsible for a different part of the pipeline. Jupyter notebook is used for experiments and analysis:
- `data`: Contains subdirectories named after class label, each containing files of jpg format with images.
- `src`: Contains the source code, organized into files:
    - `research.ipynb`: Executes and analyses experiments.
    - `config.py`: Contains configuration.
    - `data.py`: Loads, splits and transforms data.
    - `eval.py`: Performs evaluation. 
    - `models.py`: Defines models.
    - `train.py`: Handles training.
    - `visualization.py`: Visualizes dataset attributes and results.

### License

This project is released under the GNU GPL v3 license. More information can be found in the `LICENSE.txt` file.

### Contact

If you have any questions or feedback, feel free to reach out via GitHub.