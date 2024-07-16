import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from unet import UNet

def test_unet():
    # Step 1: Instantiate the model
    model = UNet()

    # Step 2: Generate a sample input
    sample_input = torch.randn(1, 1, 256, 256)  # Batch size of 1, 1 channel, 256x256 image

    # Step 3: Forward pass
    output = model(sample_input)

    # Step 4: Check the output
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    assert output.shape == sample_input.shape, "The output shape is incorrect!"

    # Optionally visualize the input and output
    input_image = sample_input.squeeze().detach().numpy()
    output_image = output.squeeze().detach().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(input_image, cmap='gray')
    axes[0].set_title('Input Image')
    axes[0].axis('off')

    axes[1].imshow(output_image, cmap='gray')
    axes[1].set_title('Output Image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_unet()
