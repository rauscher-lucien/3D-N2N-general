
import os
import warnings
import glob
import random
import torch
import numpy as np
import tifffile
import pickle
import matplotlib.pyplot as plt


def create_result_dir(project_dir, project_name, hyperparameters, train_data_dir):
    # Create a name based on project name, training directory, and hyperparameters
    base_name = os.path.basename(train_data_dir)
    hyperparams_str = '_'.join([f"{key}{value}" for key, value in hyperparameters.items()])
    name = f"{project_name}_{base_name}_{hyperparams_str}"

    results_dir = os.path.join(project_dir, name, 'results')
    os.makedirs(results_dir, exist_ok=True)
    checkpoints_dir = os.path.join(project_dir, name, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)

    return results_dir, checkpoints_dir

def create_train_dir(results_dir):
    train_dir = os.path.join(results_dir, 'train')
    os.makedirs(train_dir, exist_ok=True)

    return train_dir


def normalize_dataset(dataset):
    all_means = []
    all_stds = []
    all_sizes = []

    # Compute mean, std, and size for each stack
    for stack in dataset:
        all_means.append(np.mean(stack))
        all_stds.append(np.std(stack))
        all_sizes.append(stack.size)

    # Convert lists to numpy arrays for easier computation
    array_means = np.array(all_means)
    array_stds = np.array(all_stds)
    array_sizes = np.array(all_sizes)

    # Compute weighted average of mean and std based on array sizes
    total_size = np.sum(array_sizes)
    weighted_mean = np.sum(array_means * array_sizes) / total_size
    weighted_std = np.sqrt(np.sum(array_stds**2 * array_sizes) / total_size)

    # Set global mean and std
    mean = weighted_mean
    std = weighted_std

    # Compute global minimum and maximum over the entire dataset
    global_min = np.min([np.min(stack) for stack in dataset])
    global_max = np.max([np.max(stack) for stack in dataset])

    # Apply global normalization to the entire dataset using the global min and max
    normalized_dataset = []
    for stack in dataset:
        # Normalize each slice in the stack using the global mean and std
        stack_normalized = (stack - mean) / std

        # Normalize each slice in the stack using the global min and max
        stack_normalized = (stack - global_min) / (global_max - global_min)

        # Clip and normalize to [0, 1] for each slice in the stack using the global min and max
        stack_normalized = np.clip(stack_normalized, 0, 1)

        normalized_dataset.append(stack_normalized.astype(np.float32))

    return normalized_dataset


def compute_global_mean_and_std(dataset_path, checkpoints_path):
    """
    Computes and saves the global mean and standard deviation across all TIFF stacks
    in the given directory and its subdirectories, saving the results in the same directory.

    Parameters:
    - dataset_path: Path to the directory containing the TIFF files.
    """
    # Define the save_path in the same directory as the dataset
    save_path = os.path.join(checkpoints_path, 'normalization_params.pkl')

    # Check if the normalization_params.pkl file already exists
    if os.path.exists(save_path):
        # Load the existing mean and std values
        with open(save_path, 'rb') as f:
            params = pickle.load(f)
            global_mean = params['mean']
            global_std = params['std']
        print(f"Loaded global mean and std parameters from {save_path}")
    else:
        all_means = []
        all_stds = []
        for subdir, _, files in os.walk(dataset_path):
            for filename in files:
                if filename.lower().endswith(('.tif', '.tiff')):
                    filepath = os.path.join(subdir, filename)
                    stack = tifffile.imread(filepath)
                    all_means.append(np.mean(stack))
                    all_stds.append(np.std(stack))
                    
        global_mean = np.mean(all_means)
        global_std = np.mean(all_stds)
        
        # Save the computed global mean and standard deviation to a file
        with open(save_path, 'wb') as f:
            pickle.dump({'mean': global_mean, 'std': global_std}, f)
        
        print(f"Global mean and std parameters saved to {save_path}")

    return global_mean, global_std



from concurrent.futures import ProcessPoolExecutor

def process_image(file_path):
    """
    Function to read an image file and compute its mean and standard deviation.
    """
    if file_path.lower().endswith('.tiff'):
        stack = tifffile.imread(file_path)
        return np.mean(stack), np.std(stack)
    return None




def denormalize_image(normalized_img, mean, std):
    """
    Denormalizes an image back to its original range using the provided mean and standard deviation.

    Parameters:
    - normalized_img: The image to be denormalized.
    - mean: The mean used for the initial normalization.
    - std: The standard deviation used for the initial normalization.

    Returns:
    - The denormalized image.
    """
    original_img = (normalized_img * std) + mean
    return original_img.astype(np.float32)


import os
import pickle

def load_normalization_params(data_dir):
    """
    Loads the mean and standard deviation values from a pickle file located in the specified data directory.

    Parameters:
    - data_dir: Path to the directory containing the 'normalization_params.pkl' file.

    Returns:
    - A tuple containing the mean and standard deviation values.
    """
    # Construct the path to the pickle file
    load_path = os.path.join(data_dir, 'normalization_params.pkl')
    
    # Load the parameters from the pickle file
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    
    mean = params['mean']
    std = params['std']
    
    return mean, std




def load_min_max_params(data_dir):
    """
    Loads the global minimum and maximum values from a pickle file located in the specified data directory.

    Parameters:
    - data_dir: Path to the directory containing the 'min_max_params.pkl' file.

    Returns:
    - A tuple containing the global minimum and maximum values.
    """
    # Construct the path to the pickle file
    load_path = os.path.join(data_dir, 'min_max_params.pkl')
    
    # Load the parameters from the pickle file
    with open(load_path, 'rb') as f:
        params = pickle.load(f)
    
    global_min = params['global_min']
    global_max = params['global_max']
    
    return global_min, global_max




def plot_intensity_distribution(image_array, block_execution=True):
    """
    Plots the intensity distribution and controls execution flow based on 'block_execution'.
    """
    # Create a new figure for each plot to avoid conflicts
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(image_array.flatten(), bins=50, color='blue', alpha=0.7)
    ax.set_title('Intensity Distribution')
    ax.set_xlabel('Intensity Value')
    ax.set_ylabel('Frequency')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    
    if block_execution:
        plt.show()
    else:
        plt.draw()
        plt.pause(1)  # Allows GUI to update
        plt.close(fig)  # Close the new figure explicitly


def get_file_path(local_path, remote_path):

    path = ''

    # Detect the operating system
    if os.name == 'nt':  # Windows
        path = local_path
    else:  # Linux and others
        path = remote_path
    
    if not os.path.exists(path):
        warnings.warn(f"Project directory '{path}' not found. Please verify the path.")
        return
    print(f"Using file path: {path}")

    return path


def clip_extremes(data, lower_percentile=0, upper_percentile=100):
    """
    Clip pixel values to the specified lower and upper percentiles.
    """
    lower_bound = np.percentile(data, lower_percentile)
    upper_bound = np.percentile(data, upper_percentile)
    return np.clip(data, lower_bound, upper_bound)



def compute_global_min_max_and_save(dataset_path):
    """
    Computes and saves the global minimum and maximum values across all TIFF stacks
    in the given directory and its subdirectories, saving the results in the same directory.

    Parameters:
    - dataset_path: Path to the directory containing the TIFF files.
    """
    global_min = float('inf')
    global_max = float('-inf')
    for subdir, _, files in os.walk(dataset_path):
        for filename in files:
            if filename.lower().endswith(('.tif', '.tiff')):
                filepath = os.path.join(subdir, filename)
                stack = tifffile.imread(filepath)
                stack_min = np.min(stack)
                stack_max = np.max(stack)
                global_min = min(global_min, stack_min)
                global_max = max(global_max, stack_max)
    
    # Define the save_path in the same directory as the dataset
    save_path = os.path.join(dataset_path, 'min_max_params.pkl')

    # Save the computed global minimum and maximum to a file
    with open(save_path, 'wb') as f:
        pickle.dump({'global_min': global_min, 'global_max': global_max}, f)
    
    print(f"Global min and max parameters saved to {save_path}")
    return global_min, global_max




def tensor_to_image(tensor, image_path):
    """
    Converts a PyTorch tensor to a NumPy array, normalizes it to the 0-1 range, and saves the image.

    Args:
    - tensor (torch.Tensor): A tensor representing the image.
    - image_path (str): Path to save the image.
    """
    if tensor.dim() == 4:  # Batch of images
        tensor = tensor[0]  # Take the first image from the batch
    tensor = tensor.detach().cpu().numpy()

    # Handle 3D tensor (D, H, W) by selecting the middle slice
    if tensor.shape[0] > 3:  # Assuming the first dimension is depth
        tensor = tensor[tensor.shape[0] // 2]  # Select the middle slice

    if tensor.ndim == 3 and tensor.shape[0] == 3:  # RGB image
        img = np.transpose(tensor, (1, 2, 0))  # Convert to (H, W, C)
    else:  # Grayscale image or single slice from 3D volume
        img = np.squeeze(tensor)  # Remove channel dim if it exists

    img = img - np.min(img)
    img = img / np.max(img)

    plt.imshow(img, cmap='gray' if img.ndim == 2 else None)
    plt.axis('off')
    plt.savefig(image_path)
    plt.close()  # Close the figure to avoid memory leak


def print_tiff_filenames(root_folder_path):
    """
    Prints the filenames of TIFF files in the specified folder and its subdirectories.
    
    Parameters:
    - root_folder_path: Path to the root folder containing TIFF stack files.
    """
    for subdir, _, files in os.walk(root_folder_path):
        sorted_files = sorted([f for f in files if f.lower().endswith(('.tif', '.tiff'))])
        for filename in sorted_files:
            print(filename)


def get_device():
    if torch.cuda.is_available():
        print("GPU is available")
        device = torch.device("cuda:0")
    else:
        print("GPU is not available")
        device = torch.device("cpu")
    
    return device


def plot_intensity_line_distribution(image, title='1', bins=200):
    plt.figure(figsize=(10, 5))

    if isinstance(image, torch.Tensor):
        # Ensure it's on the CPU and convert to NumPy
        image = image.detach().numpy()

    # Use numpy.histogram to bin the pixel intensity data, using the global min and max
    intensity_values, bin_edges = np.histogram(image, bins=bins, range=(np.min(image), np.max(image)))
    # Calculate bin centers from edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    plt.plot(bin_centers, intensity_values, label='Pixel Intensity Distribution')
    
    plt.title('Pixel Intensity Distribution ' + title)
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_as_image(data, title='Image Display', cmap='gray', colorbar=True):
    plt.figure(figsize=(6, 6))

    if isinstance(data, torch.Tensor):
        # Ensure it's on the CPU and convert to NumPy
        data = data.detach().cpu().numpy()

    # Check if the data has multiple channels and select the first one if so
    if data.ndim == 3 and (data.shape[0] == 3 or data.shape[0] == 1):
        data = data[0]  # Assume the first channel for visualization if it's a 3-channel image

    img = plt.imshow(data, cmap=cmap)
    plt.title(title)
    plt.axis('off')  # Turn off axis numbers and ticks

    if colorbar:
        plt.colorbar(img)

    plt.show()

    print(data.min())
    print(data.max())
