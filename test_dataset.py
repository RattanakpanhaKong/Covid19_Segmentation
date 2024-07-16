# # Display only one pairs
# import os
# import matplotlib.pyplot as plt
# from dataset import XRayDataset, get_transforms
#
#
# def plot_image_and_mask(image, mask, image_filename, mask_filename):
#     fig, axes = plt.subplots(1, 2, figsize=(10, 5))
#
#     axes[0].imshow(image.squeeze(), cmap='gray')
#     axes[0].set_title(f'Image: {image_filename}')
#     axes[0].axis('off')
#
#     axes[1].imshow(mask.squeeze(), cmap='gray')
#     axes[1].set_title(f'Mask: {mask_filename}')
#     axes[1].axis('off')
#
#     plt.tight_layout()
#     plt.show()
#
#
# def test_dataset():
#     image_dir = 'data/train/image'  # Replace with your actual path
#     mask_dir = 'data/train/mask'  # Replace with your actual path
#
#     dataset = XRayDataset(image_dir, mask_dir, transform=get_transforms())
#
#     # Load a sample image and mask
#     image, mask = dataset[0]
#
#     # Get the corresponding file names
#     image_filename = dataset.images[0]
#     mask_filename = dataset.images[0]
#
#     # Convert tensors to numpy arrays for plotting
#     image_np = image.numpy()
#     mask_np = mask.numpy()
#
#     plot_image_and_mask(image_np, mask_np, image_filename, mask_filename)
#
#
# if __name__ == "__main__":
#     test_dataset()

# # Display multiple pairs
import matplotlib.pyplot as plt
from dataset import XRayDataset, get_transforms


def plot_images_and_masks(images, masks, filenames):
    num_images = len(images)
    fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))

    for i in range(num_images):
        image, mask = images[i], masks[i]
        image_filename, mask_filename = filenames[i]

        axes[i, 0].imshow(image.squeeze(), cmap='gray')
        axes[i, 0].set_title(f'Image: {image_filename}')
        axes[i, 0].axis('off')

        axes[i, 1].imshow(mask.squeeze(), cmap='gray')
        axes[i, 1].set_title(f'Mask: {mask_filename}')
        axes[i, 1].axis('off')

    plt.tight_layout()
    plt.show()


def test_dataset():
    image_dir = 'data/train/image'  # Replace with your actual path
    mask_dir = 'data/train/mask'  # Replace with your actual path

    dataset = XRayDataset(image_dir, mask_dir, transform=get_transforms())

    num_samples = 3  # Number of samples to display
    images, masks, filenames = [], [], []

    for i in range(num_samples):
        image, mask = dataset[i]
        image_filename = dataset.images[i]
        mask_filename = dataset.images[i]

        images.append(image.numpy())
        masks.append(mask.numpy())
        filenames.append((image_filename, mask_filename))

    plot_images_and_masks(images, masks, filenames)


if __name__ == "__main__":
    test_dataset()




# # Display multiple pairs till end of images
# import matplotlib.pyplot as plt
# from dataset import XRayDataset, get_transforms
#
#
# def plot_images_and_masks(images, masks, filenames):
#     num_images = len(images)
#     fig, axes = plt.subplots(num_images, 2, figsize=(10, 5 * num_images))
#
#     for i in range(num_images):
#         image, mask = images[i], masks[i]
#         image_filename, mask_filename = filenames[i]
#
#         axes[i, 0].imshow(image.squeeze(), cmap='gray')
#         axes[i, 0].set_title(f'Image: {image_filename}')
#         axes[i, 0].axis('off')
#
#         axes[i, 1].imshow(mask.squeeze(), cmap='gray')
#         axes[i, 1].set_title(f'Mask: {mask_filename}')
#         axes[i, 1].axis('off')
#
#     plt.tight_layout()
#     plt.show()
#
#
# def test_dataset():
#     image_dir = 'data/train/image'  # Replace with your actual path
#     mask_dir = 'data/train/mask'  # Replace with your actual path
#
#     dataset = XRayDataset(image_dir, mask_dir, transform=get_transforms())
#
#     num_samples = len(dataset)  # Number of samples to display (all images in the dataset)
#     images, masks, filenames = [], [], []
#
#     for i in range(num_samples):
#         image, mask = dataset[i]
#         image_filename = dataset.images[i]
#         mask_filename = dataset.images[i]
#
#         images.append(image.numpy())
#         masks.append(mask.numpy())
#         filenames.append((image_filename, mask_filename))
#
#         # Plot in batches of 5
#         if (i + 1) % 5 == 0 or i == num_samples - 1:
#             plot_images_and_masks(images, masks, filenames)
#             images, masks, filenames = [], [], []  # Reset lists for the next batch
#
#
# if __name__ == "__main__":
#     test_dataset()