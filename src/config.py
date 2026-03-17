class BaseConfig:
    """
    Base configuration for training and data processing.

    Contains global constants used across the project, including:
    - dataset splitting ratios
    - training hyperparameters
    - reproducibility settings

    Attributes:
    - SEED (int): Random seed for reproducibility.
    - BATCH_SIZE (int): Number of samples per batch.
    - TRAIN_SIZE (float): Proportion of data used for training.
    - VAL_SIZE (float): Proportion of data used for validation.
    - TEST_SIZE (float): Proportion of data used for testing.
    - LR (float): Learning rate for optimizer.
    - MOMENTUM (float): Momentum value (used in optimizers like SGD).
    - EPOCHS (int): Number of training epochs.
    """
    SEED = 42
    BATCH_SIZE = 16
    TRAIN_SIZE = 0.7
    VAL_SIZE = 0.15
    TEST_SIZE = 0.15
    LR = 0.001
    MOMENTUM = 0.9
    EPOCHS = 10