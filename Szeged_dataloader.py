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

class Szeged_Dataset(Dataset):
    def __init__(self, mode="train", input_size=(options.img_w, options.img_h)):
        # self.imgs_dir = options.dataset  # Path to AMR dataset
        if options.datamode:
            self.imgs_dir = options.dataset + '/HULA/ABMR_dataset/multi/'
        else:
            self.imgs_dir = options.dataset + '/HULA/ABMR_dataset/AMR_raw_gloms/'
        self.mode = mode
        self.input_size = input_size
        self.label_code = {"Non-AMR": 0, "AMR": 1, "Inconclusive": 2, "None": 3}
        self.patients = {}
        self.folds = options.dataset + '/HULA/ABMR_dataset/folds/'

        glom_files = []

        if self.mode == "train":
            set = list(range(5))
            set.remove(options.test_fold_val)
            for i in set:
                glom_files = glom_files + np.load(self.folds + 'amr_fold{}.npz'.format(i))['FILES'].tolist()
            print('Training on 4 folds, image count:', len(glom_files))
        elif self.mode == "val":
            glom_files = np.load(self.folds + 'amr_fold{}.npz'.format(options.test_fold_val))['FILES'].tolist()
            print("Testing image count:", len(glom_files))

        # Create patient dictionary
        for file in glom_files:
            if '.scn' in file:
                file = file.split('/')[-1].split('.scn')[0]
            elif '.ndpi' in file:
                file = file.split('/')[-1].split('.ndpi')[0]
            else:
                patient_name_list = file.split('-')
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
        if options.datamode:
            full_files = [f for f in os.listdir(self.imgs_dir[:-1]) if os.path.isfile(os.path.join(self.imgs_dir[:-1], f))]
        # getitem() builds one stack for a single patient, patient indexed by given index
        existing_patient_images = self.patients[list(self.patients.keys())[index]]
        patient_samples = []
        # Sample options.stack_size images from this patient
        target_label = 'AMR' if existing_patient_images[0][0] == 'A' else 'Non-AMR'
        if len(existing_patient_images) < options.stack_size:
            # Take all existing images, shuffle to randomize order
            random.shuffle(existing_patient_images)
            for img_name in existing_patient_images:
                if options.datamode:
                    matching = [s for s in full_files if img_name[:-4] in s]
                    img = Image.open(self.imgs_dir + matching[0])
                else:
                    img = Image.open(self.imgs_dir + img_name)
                img = transforms.ToTensor()(img)
                img = (img - img.min()) / (img.max() - img.min())
                if self.mode == 'train':
                    # normalization & augmentation
                    img = transforms.Resize(self.input_size, Image.BILINEAR)(img)
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
                if options.datamode:
                    matching = [s for s in full_files if file_name[:-4] in s]
                    img = Image.open(self.imgs_dir + matching[0])
                else:
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
            for _ in range(options.stack_size):
                # Sampling without replacement
                original = random.sample(existing_patient_images, 1)
                existing_patient_images.remove(original[0])
                file_name = original[0]

                if options.datamode:
                    matching = [s for s in full_files if file_name[:-4] in s]
                    img = Image.open(self.imgs_dir + matching[0])
                else:
                    img = Image.open(self.imgs_dir + file_name)
                img = transforms.ToTensor()(img)
                img = (img - img.min()) / (img.max() - img.min())

                if self.mode == 'train':
                    # normalization & augmentation
                    img = transforms.Resize(self.input_size, Image.BILINEAR)(img)
                    img = transforms.RandomHorizontalFlip()(img)
                    img = transforms.RandomVerticalFlip()(img)
                    # Classifier transforms in main script...

                if self.mode == 'val':
                    img = transforms.Resize(self.input_size, Image.BILINEAR)(img)

                if self.mode == 'test':
                    img = transforms.Resize(self.input_size, Image.BILINEAR)(img)
                patient_samples.append(img)

        return torch.stack(patient_samples, dim=0), self.label_code[target_label]
