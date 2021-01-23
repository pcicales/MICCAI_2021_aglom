import os
import re
import random
from PIL import Image
from random import randint

import torch
from torchvision import transforms
from ImageTransformationNetwork import TransformerNet, SEStyEncNet
from config import options
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "{}".format(options.gpu_used)
device = torch.device("cuda" if options.cuda else "cpu")


# This script loads in a stylizer model and stylizes an image using a random style


def save_image(filename, data):
    img = data.clone().clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype("uint8")
    img = Image.fromarray(img)
    img.save(filename)


# Load content image to stylize
content_image = Image.open(options.stylize_content_image)

# Some basic transformations for image
img_transform = transforms.Compose([
    transforms.Resize((options.img_h, options.img_w), Image.BILINEAR),
    transforms.ToTensor(),
    # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
content_image = img_transform(content_image)
content_image = content_image.unsqueeze(0).to(device)  # Simulate batch size, network expects [X, 3, img_h, img_w]

with torch.no_grad():
    image_transform_net = SEStyEncNet(options.num_styles).to(device)
    state_dict = torch.load(options.stylize_model)
    # Pytorch - remove saved deprecated running_* keys in InstanceNormalization from the checkpoint
    for k in list(state_dict.keys()):
        if re.search(r'in\d+\.running_(mean|var)$', k):
            del state_dict[k]

    # Load saved weights into initialized stylizer model
    image_transform_net.load_state_dict(state_dict)
    # Send stylizer to gpu if available
    image_transform_net.to(device)
    # Transfer style onto image
    if options.mix_styles == 1:
        # Mixing styles, need 3 different style numbers and 3 weights for linear combo
        alphas = torch.softmax(torch.tensor([random.random() * 3, random.random(), random.random()]), dim=0)
        style_numbers = list(range(options.num_styles))
        style_no1 = random.choice(style_numbers)
        style_numbers.remove(style_no1)
        style_no2 = random.choice(style_numbers)
        style_numbers.remove(style_no2)
        style_no3 = random.choice(style_numbers)
        output = image_transform_net(content_image, style_no1, style_no2, style_no3, alphas).cpu()
    else:
        # Random style number
        style_num = randint(0, options.num_styles - 1)
        print("Styling content image with style", str(style_num + 1))
        output = image_transform_net(content_image, style_num).cpu()

# Save output image
num_files = len(os.listdir(options.stylize_output_img_path))  # Num files in current directory, used to name new image
save_image("./Image_" + str(num_files + 1) + "_style_" + str(style_num + 1) + ".png", output[0] * 255.0)


# Train command
# python train_stylizer.py --save-dir ./save/img_transform_network --epochs 25 --lr 1e-2 --loo 0 --gpu-used 1

# Stylize command
# python train_stylizer.py --mode eval --output-img-path ./images/img_transform_network/20210114_221254/
# --stylize-model ./save/img_transform_network/20210114_221254/models/epoch_25_20210114_225636_1e0_1e4.model
# --num-styles 12 --gpu-used 3 --img_h 512 --img_w 512 --stylize-no 1
