import torch
import torch.nn as nn
import pandas as pd
import torch.backends.cudnn as cudnn
from ABMR_dataloader import ABMR_Dataset
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from models import *
from config import options
import seaborn as sns
from utils.eval_utils import compute_accuracy
import matplotlib.pyplot as plt
import matplotlib

def set_style():
    # This sets reasonable defaults for font size for
    # a figure that will go in a paper
    sns.set_context("paper")

    # Set the font to be serif, rather than sans
    sns.set(font='serif')

    # Make the background white, and specify the
    # specific font family
    sns.set_style("white", {
        "font.family": "serif",
        "font.serif": ["Times", "Palatino", "serif"]
    })

def test():
    print('Beginning testing...')
    net_out = []
    net_loss = []
    net.eval()
    set_style()
    with torch.no_grad():
        for i in range(options.val_iters):
            test_loss = 0
            targets, outputs = [], []
            for batch_id, (data, target) in enumerate(test_loader):
                len_batch = len(data)
                data = data.view(len_batch * options.stack_size, 3, options.img_h, options.img_w)
                if options.cuda:
                    data, target = data.cuda(), target.cuda()
                output = net(data)
                if options.num_classes == 2:
                    batch_loss = criterion(output, target.type_as(output).unsqueeze(1))
                else:
                    batch_loss = criterion(output, target)
                targets += [target]
                outputs += [output]
                test_loss += batch_loss
            net_out += [torch.cat(outputs)]
            net_loss += [test_loss]

        print("Completed runs for fold {}.".format(j))
        output_stack = torch.stack(net_out, dim=0).permute(1, 0, 2)
        loss_stack = torch.stack(net_loss)
        targets = torch.cat(targets)

        if options.num_classes == 2:
            output_stack = nn.functional.sigmoid(output_stack)
        else:
            output_stack = torch.softmax(output_stack, dim=2)  # Row-wise softmax

        out_mean = torch.mean(output_stack, dim=1)
        loss_mean = torch.mean(loss_stack, dim=0)
        test_acc = compute_accuracy(targets, out_mean, options.num_classes)
        print('Validation set accuracy for fold {}: {}'.format(j, test_acc))
        print('Validation set loss for fold {}: {}'.format(j, loss_mean))
        patient_name_list = list(test_loader.dataset.patients.keys())
        for i in range(len(test_loader.dataset)):
            if patient_name_list[i][0] == 'A':
                gt = 'ABMR'
            else:
                gt = 'Non-ABMR'
            if out_mean[i].item() > 0.5:
                model_prediction = 'ABMR'
            else:
                model_prediction = 'Non-ABMR'
            prediction_var = torch.var(output_stack[i])
            plt.style.use(['seaborn-white', 'seaborn-paper'])
            sns.kdeplot(data=output_stack[i].detach().cpu().numpy(), clip=[0, 1], shade=True)
            sns.set_palette('gist_heat')
            plt.title("GT: " + gt + ", Prediction: " + model_prediction + ", Var: " + str(round(prediction_var.item(), 3)))
            plt.xlabel("ABMR Probability")
            plt.xlim(0, 1)
            ax = plt.gca()
            ax.legend_ = None
            plt.savefig("./save/density_plots/density_plot_" + patient_name_list[i] + ".png", dpi=300)
            plt.clf()


if __name__ == '__main__':
    ##################################
    # Creating the classifier model
    ##################################
    net = None
    if options.classifier_model == 'resnet':
        net = resnet.resnet50(pretrained=True)
        net.fc = nn.Linear(net.fc.in_features, options.num_classes)
    elif options.classifier_model == 'vgg':
        net = vgg19_bn(pretrained=True, num_classes=options.num_classes)
    elif options.classifier_model == 'inception':
        net = inception_v3(pretrained=True)
        net.aux_logits = False
        net.fc = nn.Linear(2048, options.num_classes)
    elif options.classifier_model == 'densenet':
        net = densenet.densenet121(pretrained=True, drop_rate=0.2)
        net.classifier = nn.Linear(net.classifier.in_features, out_features=options.num_classes)
    elif options.classifier_model == 'morphset':
        if options.encoder == 'resnet50':
            enc = resnet.resnet50(pretrained=True)
            enc = nn.Sequential(*(list(enc.children())[:-2]))
        elif 'efficientnet' in options.encoder:
            enc = EfficientNet.from_pretrained(options.encoder, include_top=True)
            a = list(enc.children())[0]
            b = list(enc.children())[1]
            c = list(enc.children())[2]
            cx = nn.Sequential(*list(c.children()))
            d = list(enc.children())[3]
            e = list(enc.children())[4]
            enc = nn.Sequential(*[a, b, cx, d, e])
        net = MorphSet(options.img_c, options.num_classes, enc)

    #######################################
    # Use cuda and load in pretrained model
    #######################################
    if options.cuda:
        cudnn.benchmark = True
        net.cuda()
        net = nn.DataParallel(net)

    if options.num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    # selected models, in order of validation folds (i.e. 0 -> fold 0)
    top_model_list = ['./save/20210227_143223/models/260.ckpt', './save/20210227_143236/models/260.ckpt',
                      './save/20210227_143247/models/284.ckpt']
    for j in range(len(top_model_list)):
        checkpoint = torch.load(top_model_list[j])
        state_dict = checkpoint['state_dict']
        net.load_state_dict(state_dict)
        print("Loaded model", top_model_list[j])
        test_dataset = ABMR_Dataset(mode='test', test=j)
        test_loader = DataLoader(test_dataset, batch_size=3*options.batch_size,
                         shuffle=False, num_workers=options.num_workers, drop_last=False)
        test()
