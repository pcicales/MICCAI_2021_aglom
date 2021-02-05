import os
import random
import torch
from torchvision import transforms
from config import options
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(options.gpu_used)
device = torch.device("cuda" if options.cuda else "cpu")


class MICCAIDataset(Dataset):
    def __init__(self, mode="train", input_size=(256, 256)):
        # self.imgs_dir = options.dataset  # Path to AMR dataset
        self.imgs_dir = '/home/cougarnet.uh.edu/pcicales/Documents/data/ABMR_dataset/AMR_raw_gloms/'
        self.mode = mode
        self.input_size = input_size
        self.label_code = {"Non-AMR": 0, "AMR": 1, "Inconclusive": 2, "None": 3}
        self.patients = {}

        glom_files = []

        if self.mode == "train":
            set = list(range(5))
            set.remove(options.test_fold_val)
            for i in set:
                glom_files = glom_files + np.load(options.data_folds + 'amr_fold{}.npz'.format(i))['FILES'].tolist()
            print('Training on 4 folds, image count:', len(glom_files))
        elif self.mode == "test":
            glom_files = np.load(options.data_folds + 'amr_fold{}.npz'.format(options.test_fold_val))['FILES'].tolist()
            print("Testing image count:", len(glom_files))
        #     if '.scn' in ALL_IMG[i]:
        #         SLIDE_ID[i] = ALL_IMG[i].split('/')[-1].split('.scn')[0]
        #     elif '.ndpi' in ALL_IMG[i]:
        #         SLIDE_ID[i] = ALL_IMG[i].split('/')[-1].split('.ndpi')[0]
        #     else:
        #         name_list = ALL_IMG[i].split('/')[-1].split('-')
        #         if '-.jpg' in ALL_IMG[i]:
        #             SLIDE_ID[i] = '-'.join(name_list[0:-2])
        #         else:
        #             SLIDE_ID[i] = '-'.join(name_list[0:-1])
        # Create patient dictionary
        for file in glom_files:
            if '.scn' in file:
                file = file.split('/')[-1].split('.scn')[0]
            elif '.ndpi' in file:
                file = file.split('/')[-1].split('.ndpi')[0]
            else:
                patient_name_list = file.split('/')[-1].split('-')
                if '-.jpg' in file:
                    patient_name = "-".join(patient_name_list[:-2])
                else:
                    patient_name = "-".join(patient_name_list[:-1])

            if patient_name in self.patients:
                self.patients[patient_name].append(file)
            else:
                self.patients[patient_name] = [file]

    def __len__(self):
        return len(self.patients)  # Length of dataset is number of patients

    def __getitem__(self, index):
        # getitem() builds one stack for a single patient, patient indexed by given index
        existing_patient_images = self.patients[list(self.patients.keys())[index]]
        patient_samples = []
        # Sample options.stack_size images from this patient
        if len(existing_patient_images) < options.stack_size:
            # Take all existing images
            for img_name in existing_patient_images:
                img = Image.open(self.imgs_dir + img_name)
                img = transforms.ToTensor()(img)
                img = (img - img.min()) / (img.max() - img.min())
                if self.mode == 'train':
                    # normalization & augmentation
                    img = transforms.Resize(self.input_size, Image.BILINEAR)(img)  # Bilinear resizing?
                    img = transforms.RandomHorizontalFlip()(img)
                    img = transforms.RandomVerticalFlip()(img)
                    # Classifier transforms in main script...

                if self.mode == 'val':
                    img = transforms.Resize(self.input_size, Image.BILINEAR)(img)

                if self.mode == 'test':
                    img = transforms.Resize(self.input_size, Image.BILINEAR)(img)

                patient_samples.append(img)

            for i in range(options.stack_size - len(existing_patient_images)):
                # Sampling with replacement, ToDo: Add in option for sampling without replacement
                sampled_img = random.sample(existing_patient_images, 1)
                # existing_files.remove(original[0])
                file_name = sampled_img[0]

                img = Image.open(self.imgs_dir + file_name)
                img = transforms.ToTensor()(img)
                img = (img - img.min()) / (img.max() - img.min())

                if self.mode == 'train':
                    # normalization & augmentation
                    img = transforms.Resize(self.input_size, Image.BILINEAR)(img)  # Bilinear resizing?
                    img = transforms.RandomHorizontalFlip()(img)
                    img = transforms.RandomVerticalFlip()(img)
                    # Classifier transforms in main script...

                if self.mode == 'val':
                    img = transforms.Resize(self.input_size, Image.BILINEAR)(img)

                if self.mode == 'test':
                    img = transforms.Resize(self.input_size, Image.BILINEAR)(img)

                patient_samples.append(img)
        else:
            # Patient has more images than options.stack_size, sample stack_size images from the vector
            for i in range(options.stack_size):
                # Sampling without replacement
                original = random.sample(existing_patient_images, 1)
                existing_patient_images.remove(original[0])
                file_name = original[0]

                img = Image.open(self.imgs_dir + file_name)
                img = transforms.ToTensor()(img)
                img = (img - img.min()) / (img.max() - img.min())

                if self.mode == 'train':
                    # normalization & augmentation
                    img = transforms.Resize(self.input_size, Image.BILINEAR)(img)  # Bilinear resizing?
                    img = transforms.RandomHorizontalFlip()(img)
                    img = transforms.RandomVerticalFlip()(img)
                    # Classifier transforms in main script...

                if self.mode == 'val':
                    img = transforms.Resize(self.input_size, Image.BILINEAR)(img)

                if self.mode == 'test':
                    img = transforms.Resize(self.input_size, Image.BILINEAR)(img)
                patient_samples.append(img)

        target_label = 'AMR' if existing_patient_images[0][0] == 'A' else 'Non-AMR'

        for temp in patient_samples:
            if isinstance(temp, str):
                print('Something bad happened.')

        return torch.stack(patient_samples, dim=0), self.label_code[target_label]
