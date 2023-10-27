import numpy as np
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from PIL import Image
import os
import random
import matplotlib.pyplot as plt
import hashlib


# function to resize images and apply light augmentation
def get_transform_options():
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
    ])
    return transform


# function to load the dataset and apply transformation of images like resizing and light augmentation.
# The output is the transformed dataset
def load_dataset(transform_options, directory_path):
    if os.path.exists(directory_path):
        dataset = ImageFolder(root=directory_path, transform=transform_options)
        return dataset
    else:
        print(f"Directory not found: {directory_path}")
        return 0


# Function to convert the images to grey scale and label the images.
# The images are labelled starting from index 0.
# If the image name starts with 0 - angry class, 1- neutral class, 2- engaged, 3- bored.
def clean_save_images(output_directory, data_loader):
    os.makedirs(output_directory, exist_ok=True)
    image_count = 0
    for batch in data_loader:
        images, labels = batch
        for i, (image, label) in enumerate(zip(images, labels)):
            grayscale_image = transforms.ToPILImage()(image[0])
            grayscale_image.save(f'{output_directory}/{label}_{image_count}.jpg')
            image_count += 1


# Function that calculates the total number of images in each class adn assigning the labels to the bar plot.

def get_class_distribution(path):
    angry = 0
    neutral = 0
    engaged = 0
    bored = 0
    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            class_label = int(filename.split("_")[0])
            if class_label == 0:
                angry += 1
            elif class_label == 1:
                neutral += 1
            elif class_label == 2:
                engaged += 1
            elif class_label == 3:
                bored += 1
    class_labels = ['Class 0 - Anger', 'Class 1 - Neutral', 'Class 2 - Engaged', 'Class 3 - Bored']
    class_counts = [angry, neutral, engaged, bored]
    return class_labels, class_counts


# Function to plot barchart to show distribution of images in each class.

def plot_barchart(class_labels, class_counts):
    plt.figure(figsize=(8, 4))
    plt.bar(class_labels, class_counts, color='blue')
    plt.xlabel('Class Labels')
    plt.ylabel('Number of Images')
    plt.title('Number of Images in Each Class')
    plt.savefig('class_distribution.png')


# Function to generate random images from the dataset for 5*5 grid.

def get_random_images(image_directory, count):
    images = os.listdir(image_directory)
    random_images = random.sample(images, count)
    return random_images

# Function to generate the 5*5 image grid from sample images.


def image_grid(image_directory, rows, columns, images):
    fig, axes = plt.subplots(rows, columns, figsize=(10, 10))
    for i in range(rows):
        for j in range(columns):
            img_index = i * columns + j
            image_path = os.path.join(image_directory, images[img_index])
            img = Image.open(image_path)
            axes[i, j].imshow(img)
            axes[i, j].imshow(img, cmap='gray')
            axes[i, j].set_title(f"Image {img_index}")
            axes[i, j].axis('off')
    plt.tight_layout()
    plt.savefig('5x5_grid.png')

# Function to plot histogram to demonstrate the pixel intensity of the sample images.


def plot_histogram(image_directory, rows, columns, images):
    fig, axes = plt.subplots(rows, columns, figsize=(12, 12))
    for i in range(rows):
        for j in range(columns):
            img_index = i * columns + j
            image_file = images[img_index]
            image_path = os.path.join(image_directory, image_file)
            img = Image.open(image_path)
            img_array = np.array(img)
            pixel_values = img_array.ravel()
            axes[i, j].hist(pixel_values, bins=256, range=(0, 256), density=True, color='gray', alpha=0.7)
            axes[i, j].set_title(f"Image {img_index}")
            axes[i, j].set_xlabel('Pixel Intensity')
            axes[i, j].set_ylabel('Frequency')
    plt.tight_layout()
    plt.savefig('histogram pixel intensity')
    plt.show()


# Function to get the image hash to find out the duplicate images.

def get_image_hash(image_path, hash_size=8):
    image = Image.open(image_path)
    image = image.convert("L")
    image = image.resize((hash_size, hash_size), Image.BILINEAR)
    pixel_data = list(image.getdata())
    pixels = np.array(pixel_data)
    avg_pixel = pixels.mean()
    diff = pixels > avg_pixel
    return sum([2 ** i for (i, v) in enumerate(diff) if v])

# Function to remove duplicate images from the dataset.


def remove_duplicates(image_directory):
    image_hashes = {}
    duplicate_images = []
    for filename in os.listdir(image_directory):
        if filename.endswith(".jpg"):
            image_path = os.path.join(image_directory, filename)
            image_hash = get_image_hash(image_path)
            if image_hash not in image_hashes:
                image_hashes[image_hash] = image_path
            else:
                duplicate_images.append(image_path)
    for duplicate_image in duplicate_images:
        os.remove(duplicate_image)
        print(f"Removed duplicate image: {duplicate_image}")

# Function to preprocess data where are all the functions are called.


def preprocess_data():
    dataset_path = '/Users/user/Desktop/AIdatasets'
    new_dir = 'resized dataset'
    transform_options = get_transform_options()
    dataset = load_dataset(transform_options, dataset_path)
    if dataset != 0:
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        clean_save_images(new_dir, data_loader)
        labels, counts = get_class_distribution(new_dir)
        plot_barchart(labels, counts)
        random_images = get_random_images(new_dir, 5 * 5)
        image_grid(new_dir, 5, 5, random_images)
        plot_histogram(new_dir, 5, 5, random_images)
        remove_duplicates(new_dir)

# Main.


if __name__ == "__main__":
    preprocess_data()
