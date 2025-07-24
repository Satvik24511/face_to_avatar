# CycleGAN: Face-to-Avatar Translation

This repository contains a PyTorch implementation of the CycleGAN algorithm for unpaired image-to-image translation. This project is set up to translate images of human faces into avatar-style images.

## How it Works

The core idea is to learn a mapping `G: X -> Y` that translates an image from a source domain `X` (e.g., real faces) to a target domain `Y` (e.g., avatars). Since we don't have paired images, a key challenge is to ensure the translation preserves the content of the original image.

CycleGAN solves this by simultaneously training a second generator `F: Y -> X` and enforcing "cycle consistency." This means that if we translate an image from `X` to `Y` and then back to `X`, we should get the original image back (i.e., `F(G(X)) ≈ X`). A similar constraint is applied for the reverse direction (`G(F(Y)) ≈ Y`).

Adversarial discriminators are used to ensure that the generated images in each domain are indistinguishable from real images.

## Project Structure

The project is organized as follows:

-   `config.py`: All hyperparameters and configuration settings.
-   `dataset.py`: `ImageDataset` class for loading data.
-   `generator_model.py`: The generator network architecture (using residual blocks).
-   `discriminator_model.py`: The discriminator network architecture (PatchGAN).
-   `train.py`: The main script for training the models.
-   `utils.py`: Helper functions for saving/loading checkpoints and images.
-   `face_to_avatar/`: The dataset directory.
-   `saved_images/`: Sample images generated during training are saved here.
-   `runs/`: Directory for TensorBoard logs.

## Getting Started

### Prerequisites

Make sure you have the following libraries installed:

-   PyTorch
-   Albumentations (for data augmentation)
-   Pillow
-   NumPy
-   tqdm

You can install them using pip:

```bash
pip install torch torchvision albumentations pillow numpy tqdm
```

### Dataset

1.  Create a directory (e.g., `face_to_avatar`).
2.  Inside, create four subdirectories: `trainA`, `trainB`, `testA`, and `testB`.
3.  Place your source domain images (e.g., faces) in `trainA` and `testA`.
4.  Place your target domain images (e.g., avatars) in `trainB` and `testB`.
5.  Update the `TRAIN_DIR` and `VAL_DIR` variables in `config.py` if you use a different directory name.

### Training

To start training, simply run the `train.py` script:

```bash
python train.py
```

You can customize the training process by modifying the hyperparameters in `config.py`.

### Monitoring and Results

-   **TensorBoard:** You can monitor the training progress (losses, generated images) using TensorBoard. Point it to the `runs/cyclegan` directory.
-   **Generated Images:** During training, sample translated images will be saved in the `saved_images` directory, allowing you to see the model's progress.
-   **Checkpoints:** Model checkpoints are saved at the end of each epoch, so you can resume training later.

## Acknowledgments

This work is based on the original CycleGAN paper:

[Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) by Jun-Yan Zhu, Taesung Park, Phillip Isola, and Alexei A. Efros.
