import os
import numpy as np
import torch
import tifffile

from skimage import filters, exposure
import matplotlib.pyplot as plt
from PIL import Image

from utils import *

class VolumeSubstackDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, stack_depth=32, transform=None):
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.stack_depth = stack_depth
        self.preloaded_data = {}
        self.pairs = self.preload_volumes(root_folder_path)

    def preload_volumes(self, root_folder_path):
        pairs = []
        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith('.tiff')])
            for f in sorted_files:
                full_path = os.path.join(subdir, f)
                volume = tifffile.imread(full_path)
                self.preloaded_data[full_path] = volume
                num_slices = volume.shape[0]
                # Store indices for the start of each possible substack
                total_possible_stacks = (num_slices - 2 * self.stack_depth) // 2 + 1
                for i in range(total_possible_stacks):
                    start_index = 2 * i  # Start index of the first slice in the substack
                    pairs.append((full_path, start_index))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        file_path, start_index = self.pairs[index]
        
        # Access the preloaded entire volume
        volume = self.preloaded_data[file_path]
        
        # Determine indices for the input and target substacks
        input_indices = range(start_index, start_index + 2 * self.stack_depth, 2)
        target_indices = range(start_index + 1, start_index + 1 + 2 * self.stack_depth, 2)
        
        # Fetch the actual slices
        input_stack = volume[input_indices]
        target_stack = volume[target_indices]

        if self.transform:
            input_stack, target_stack = self.transform((input_stack, target_stack))

        input_stack = input_stack[np.newaxis, ...]
        target_stack = target_stack[np.newaxis, ...]

        return input_stack, target_stack



class ValidationDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, stack_depth=32, transform=None):
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.stack_depth = stack_depth
        self.substack_depth = 2 * stack_depth  # 64 slices: 32 input + 32 target
        self.preloaded_data = {}
        self.pairs = self.preload_volumes(root_folder_path)

    def preload_volumes(self, root_folder_path):
        pairs = []
        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith('.tiff')])
            for f in sorted_files:
                full_path = os.path.join(subdir, f)
                volume = tifffile.imread(full_path)
                self.preloaded_data[full_path] = volume
                num_slices = volume.shape[0]

                # Divide volume into non-overlapping substacks
                for i in range(0, num_slices - self.substack_depth + 1, self.substack_depth):
                    pairs.append((full_path, i))
                
                # Handle remaining slices if they do not fit into a full substack
                if num_slices % self.substack_depth != 0:
                    pairs.append((full_path, num_slices - self.substack_depth))

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        file_path, start_index = self.pairs[index]
        
        # Access the preloaded entire volume
        volume = self.preloaded_data[file_path]
        
        # Determine indices for the input and target substacks
        input_indices = range(start_index, start_index + self.substack_depth, 2)
        target_indices = range(start_index + 1, start_index + self.substack_depth, 2)
        
        # Fetch the actual slices
        input_stack = volume[input_indices]
        target_stack = volume[target_indices]

        if self.transform:
            input_stack, target_stack = self.transform((input_stack, target_stack))

        input_stack = input_stack[np.newaxis, ...]
        target_stack = target_stack[np.newaxis, ...]

        return input_stack, target_stack



class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, root_folder_path, stack_depth=32, transform=None):
        self.root_folder_path = root_folder_path
        self.transform = transform
        self.stack_depth = stack_depth
        self.preloaded_data = {}
        self.pairs = self.preload_volumes(root_folder_path)

    def preload_volumes(self, root_folder_path):
        pairs = []
        for subdir, _, files in os.walk(root_folder_path):
            sorted_files = sorted([f for f in files if f.lower().endswith('.tiff')])
            for f in sorted_files:
                full_path = os.path.join(subdir, f)
                volume = tifffile.imread(full_path)
                self.preloaded_data[full_path] = volume
                num_slices = volume.shape[0]
                
                num_stacks = (num_slices + self.stack_depth - 1) // self.stack_depth  # Calculate the number of stacks needed
                
                for i in range(num_stacks):
                    start_index = i * self.stack_depth
                    pairs.append((full_path, start_index))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        file_path, start_index = self.pairs[index]
        
        # Access the preloaded entire volume
        volume = self.preloaded_data[file_path]
        
        # Calculate end index
        end_index = start_index + self.stack_depth
        if end_index > volume.shape[0]:
            end_index = volume.shape[0]
        
        # Fetch the actual slices
        input_stack = volume[start_index:end_index]

        # Pad the stack if necessary
        if input_stack.shape[0] < self.stack_depth:
            padding = self.stack_depth - input_stack.shape[0]
            input_stack = np.pad(input_stack, ((0, padding), (0, 0), (0, 0)), mode='reflect')
        
        if self.transform:
            input_stack = self.transform(input_stack)

        input_stack = input_stack[np.newaxis, ...]

        return input_stack