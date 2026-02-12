import sys
sys.path.insert(0, '/home/comp/csrkzhu/code/Compare/nnUNet')
import os
os.environ['nnUNet_raw'] = '/home/comp/csrkzhu/code/Compare/nnUNet/nnUNet_data/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '/home/comp/csrkzhu/code/Compare/nnUNet/nnUNet_data/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '/home/comp/csrkzhu/code/Compare/nnUNet/nnUNet_data/nnUNet_results'

# Try to mimic what the dataloader does - load one sample
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
dataset = nnUNetDataset('/home/comp/csrkzhu/code/Compare/nnUNet/nnUNet_data/nnUNet_preprocessed/Dataset501_HyperBody', 
                        case_identifiers=None, 
                        num_images_properties_loading_threshold=0)
print(f"Dataset loaded with {len(dataset)} cases")
# Try loading one item
keys = list(dataset.keys())
print(f"First key: {keys[0]}")
item = dataset.load_case(keys[0])
print(f"Loaded case successfully, data shape: {item['data'].shape}")
print("Single-process data loading works!")
