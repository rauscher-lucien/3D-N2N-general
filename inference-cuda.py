import os
import sys
import argparse
import logging
import glob
import torch
import numpy as np
import tifffile
from torchvision import transforms

sys.path.append(os.path.join(".."))

from model import *
from transforms import *
from utils import *
from dataset import *

class StreamToLogger(object):
    def __init__(self, logger, log_level=logging.INFO):
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.log_level, line.rstrip())

    def flush(self):
        pass

def setup_logging(log_file='logging.log'):
    logging.basicConfig(filename=log_file, filemode='a', format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO, datefmt='%Y-%m-%d %H:%M:%S')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console_handler)
    sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
    sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)

def load(checkpoints_dir, model, epoch=1, optimizer=None, device='cpu'):
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())  

    checkpoint_path = os.path.join(checkpoints_dir, f'best_model.pth')
    dict_net = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(dict_net['model'])
    optimizer.load_state_dict(dict_net['optimizer'])
    epoch = dict_net['epoch']

    model.to(device)

    print('Loaded %dth network' % epoch)

    return model, epoch

def main():
    setup_logging()
    
    parser = argparse.ArgumentParser(description='Process inference parameters.')
    parser.add_argument('--project_dir', type=str, help='Path to the project directory', default=None)
    parser.add_argument('--data_dir', type=str, help='Path to the data directory', default=None)
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for inference, e.g., "cuda:0" or "cpu"')

    args = parser.parse_args()

    if os.getenv('RUNNING_ON_SERVER') == 'true':
        project_dir = args.project_dir
        data_dir = args.data_dir
    else:
        project_dir = r"Z:\members\Rauscher\projects\one_adj_slice\big_data_small-no_nema-no_droso-test_1"
        data_dir = r"Z:\members\Rauscher\data\big_data_small\platynereis"

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    
    project_name = os.path.basename(project_dir)
    inference_name = os.path.basename(data_dir)
    
    results_dir = os.path.join(project_dir, 'results')
    checkpoints_dir = os.path.join(project_dir, 'checkpoints')

    inference_folder = os.path.join(results_dir, inference_name)
    os.makedirs(inference_folder, exist_ok=True)
    
    filenames = glob.glob(os.path.join(data_dir, "*.tif")) + glob.glob(os.path.join(data_dir, "*.tiff"))
    print("Following files will be denoised:  ", filenames)

    print(f"Using device: {device}")

    mean, std = load_normalization_params(checkpoints_dir)
    
    inf_transform = transforms.Compose([
        Normalize(mean, std),
        CropToMultipleOf16Inference(),
        ToTensor(),
    ])

    inv_inf_transform = transforms.Compose([
        ToNumpy(),
        Denormalize(mean, std)
    ])

    inf_dataset = InferenceDataset(
        data_dir,
        transform=inf_transform
    )

    batch_size = 8
    print("Dataset size:", len(inf_dataset))
    inf_loader = torch.utils.data.DataLoader(
        inf_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    model = NewUNet()
    model, epoch = load(checkpoints_dir, model, device=device)

    print("starting inference")
    output_images = []

    with torch.no_grad():
        model.eval()
        for batch, data in enumerate(inf_loader):
            input_img = data.to(device)
            output_img = model(input_img)
            output_img_np = inv_inf_transform(output_img)

            for img in output_img_np:
                output_images.append(img)

            print('BATCH %04d/%04d' % (batch, len(inf_loader)))
    
    output_stack = np.stack(output_images, axis=0)
    filename = f'output_stack-{project_name}-{inference_name}-epoch{epoch}.TIFF'
    tifffile.imwrite(os.path.join(inference_folder, filename), output_stack)

    print("TIFF stacks created successfully.")

if __name__ == '__main__':
    main()
