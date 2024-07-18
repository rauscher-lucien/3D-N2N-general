import numpy as np
import os
import sys
from skimage import filters, exposure
import matplotlib.pyplot as plt
from PIL import Image
import torch

class Normalize(object):
    """
    Normalize an image using mean and standard deviation.
    
    Args:
        mean (float or tuple): Mean for each channel.
        std (float or tuple): Standard deviation for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        """
        Args:
            data (tuple): Containing input and target images to be normalized.
        
        Returns:
            Tuple: Normalized input and target images.
        """
        input_img, target_img = data

        # Normalize input image
        input_normalized = (input_img - self.mean) / self.std

        # Normalize target image
        target_normalized = (target_img - self.mean) / self.std

        return input_normalized, target_normalized
    

class NormalizeInference(object):
    """
    Normalize an image using mean and standard deviation.
    
    Args:
        mean (float or tuple): Mean for each channel.
        std (float or tuple): Standard deviation for each channel.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        """
        Args:
            data (tuple): Containing input and target images to be normalized.
        
        Returns:
            Tuple: Normalized input and target images.
        """
        input_img = data

        # Normalize input image
        input_normalized = (input_img - self.mean) / self.std

        return input_normalized


class LogScaleAndNormalize(object):
    """
    Apply logarithmic scaling followed by Z-score normalization to a single-channel image.

    Args:
        mean (float): Mean of the log-scaled data.
        std (float): Standard deviation of the log-scaled data.
        epsilon (float): A small value added to the input to avoid logarithm of zero.

    """

    def __init__(self, mean, std, epsilon=1e-10):
        self.mean = mean
        self.std = std
        self.epsilon = epsilon

    def __call__(self, data):
        """
        Apply logarithmic scaling followed by Z-score normalization to a single-channel image with dimensions (1, H, W).

        Args:
            img (numpy.ndarray): Image to be transformed, expected to be in the format (1, H, W).

        Returns:
            numpy.ndarray: Transformed image.
        """

        input_img, target_img = data

        log_scaled_mean = np.log(self.mean + self.epsilon)
        log_scaled_std = np.log(self.std + self.epsilon)

        log_scaled_input_img = np.log(input_img + self.epsilon)
        log_scaled_target_img = np.log(target_img + self.epsilon)
        

        normalized_input_img = (log_scaled_input_img - log_scaled_mean) / log_scaled_std
        normalized_target_img = (log_scaled_target_img - log_scaled_mean) / log_scaled_std


        return normalized_input_img, normalized_target_img


class RandomFlip(object):

    def __call__(self, data):

        input_img, target_img = data

        if np.random.rand() > 0.5:
            input_img = np.fliplr(input_img)
            target_img = np.fliplr(target_img)

        if np.random.rand() > 0.5:
            input_img = np.flipud(input_img)
            target_img = np.flipud(target_img)

        return input_img, target_img
    

class RandomHorizontalFlip:
    def __call__(self, data):
        """
        Apply random horizontal flipping to both the input stack of slices and the target slice.
        In 50% of the cases, only horizontal flipping is applied without vertical flipping.
        
        Args:
            data (tuple): A tuple containing the input stack and the target slice.
        
        Returns:
            Tuple: Horizontally flipped input stack and target slice, if applied.
        """
        input_stack, target_slice = data

        # Apply horizontal flipping with a 50% chance
        if np.random.rand() > 0.5:
            # Flip along the width axis (axis 1), keeping the channel dimension (axis 2) intact
            input_stack = np.flip(input_stack, axis=1)
            target_slice = np.flip(target_slice, axis=1)

        # With the modified requirements, we remove the vertical flipping part
        # to ensure that only horizontal flipping is considered.

        return input_stack, target_slice




class RandomCrop:
    def __init__(self, output_size=(64, 64)):

        self.output_size = output_size

    def __call__(self, data):

        input_stack, target_stack = data

        _, h, w = input_stack.shape
        _, h, w = target_stack.shape
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        input_cropped = input_stack[:, top:top+new_h, left:left+new_w]
        target_cropped = target_stack[:, top:top+new_h, left:left+new_w]

        return input_cropped, target_cropped
    

class CropToMultipleOf16Inference(object):
    """
    Crop each slice in a stack of images to ensure their height and width are multiples of 32.
    This is particularly useful for models that require input dimensions to be divisible by certain values.
    """

    def __call__(self, data):
        """
        Args:
            stack (numpy.ndarray): Stack of images to be cropped, with shape (H, W, Num_Slices).

        Returns:
            numpy.ndarray: Stack of cropped images.
        """

        input_slice = data

        _, h, w = data.shape

        new_h = h - (h % 16)
        new_w = w - (w % 16)

        # Calculate cropping margins
        top = (h - new_h) // 2
        left = (w - new_w) // 2

        # Generate indices for cropping
        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis].astype(np.int32)
        id_x = np.arange(left, left + new_w, 1).astype(np.int32)

        input_slice_cropped = input_slice[:, id_y, id_x].squeeze()

        return input_slice_cropped
    


class CropToMultipleOf16Validation(object):
    """
    Crop the height and width of each volume in a stack of 3D images to ensure their height and width are multiples of 16.
    The depth dimension remains intact.
    """

    def __call__(self, data):
        """
        Args:
            data (tuple): Tuple containing input and target 3D arrays with shape (D, H, W).

        Returns:
            tuple: Tuple containing cropped input and target 3D arrays.
        """

        input_volume, target_volume = data
        d, h, w = input_volume.shape  # Assuming input_volume is a numpy array with shape (D, H, W)

        new_h = h - (h % 16)
        new_w = w - (w % 16)

        # Calculate cropping margins
        start_h = (h - new_h) // 2
        start_w = (w - new_w) // 2

        input_cropped = input_volume[:, start_h:start_h + new_h, start_w:start_w + new_w]
        target_cropped = target_volume[:, start_h:start_h + new_h, start_w:start_w + new_w]

        return input_cropped, target_cropped






class ToTensor(object):
    def __call__(self, data):
        def convert_image(img):
            return torch.from_numpy(img.astype(np.float32))
        return tuple(convert_image(img) for img in data)


class ToTensorInference(object):
    def __call__(self, img):
        # Convert a single image
        return torch.from_numpy(img.astype(np.float32))


class ToNumpy(object):

    def __call__(self, data):

        return data.to('cpu').detach().numpy().transpose(0, 2, 3, 4, 1)
    
    

    
class BackTo01Range(object):
    """
    Normalize a tensor to the range [0, 1] based on its own min and max values.
    """

    def __call__(self, tensor):
        """
        Args:
            tensor: A tensor with any range of values.
        
        Returns:
            A tensor normalized to the range [0, 1].
        """
        min_val = tensor.min()
        max_val = tensor.max()
        
        # Avoid division by zero in case the tensor is constant
        if (max_val - min_val).item() > 0:
            # Normalize the tensor to [0, 1] based on its dynamic range
            normalized_tensor = (tensor - min_val) / (max_val - min_val)
        else:
            # If the tensor is constant, set it to a default value, e.g., 0, or handle as needed
            normalized_tensor = tensor.clone().fill_(0)  # Here, setting all values to 0

        return normalized_tensor


class Denormalize(object):
    """
    Denormalize an image using mean and standard deviation, then convert it to 16-bit format.
    
    Args:
        mean (float or tuple): Mean for each channel.
        std (float or tuple): Standard deviation for each channel.
    """

    def __init__(self, mean, std):
        """
        Initialize with mean and standard deviation.
        
        Args:
            mean (float or tuple): Mean for each channel.
            std (float or tuple): Standard deviation for each channel.
        """
        self.mean = mean
        self.std = std
    
    def __call__(self, img):
        """
        Denormalize the image and convert it to 16-bit format.
        
        Args:
            img (numpy array): Normalized image.
        
        Returns:
            numpy array: Denormalized 16-bit image.
        """
        # Denormalize the image by reversing the normalization process
        img_denormalized = (img * self.std) + self.mean

        # Scale the image to the range [0, 65535] and convert to 16-bit unsigned integer
        img_16bit = img_denormalized.astype(np.uint16)
        
        return img_16bit
