import torch
import torch.nn as nn
import pandas as pd
import torch.backends.cudnn as cudnn
from ABMR_dataloader import ABMR_Dataset
from GN_dataloader import GN_Dataset
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

def test(mode='morphset'):
    print('Beginning testing...')
    net_out = []
    net_loss = []
    net.eval()
    set_style()
    sns.set_palette('gist_heat')
    label_code = {"PGNMID": 0, "Fibrillary": 1, "ANCA": 2, "Membranous": 3, "IgAGN": 4, 'ABMGN': 5, 'SLEGN-IV': 6, 'IAGN': 7, 'DDD': 8, 'MPGN': 9}
    label_encoder = {y:x for x,y in label_code.items()}
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

        print("Completed runs for fold {}.".format(i))
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
        print('Validation set accuracy for fold {}: {}'.format(i, test_acc))
        print('Validation set loss for fold {}: {}'.format(i, loss_mean))
        patient_name_list = list(test_loader.dataset.patients.keys())
        patient_label_list = list(test_loader.dataset.glom_labels)
        for j in range(len(test_loader.dataset)):
            gt = patient_label_list[j]
            model_prediction = label_encoder[out_mean[j].argmax().item()]
            plt.style.use(['seaborn-white', 'seaborn-paper'])
            plt.title("Prediction: " + model_prediction + ", GT: " + str(gt),  fontsize=16)
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            # for k in range(len(out_mean[j])):
            #     # prediction_std = torch.std(output_stack[j][:, k])
            palette = sns.color_palette("dark")
            g = sns.kdeplot(data=output_stack[j].detach().cpu().numpy(), clip=[0, 1], shade=True, palette=palette)
            new_labels = list(label_code.keys())
            leg = g.axes.legend_
            new_title = 'Class Distributions'
            leg.set_title(new_title)
            for t, l in zip(leg.texts, new_labels): t.set_text(l)
            leg.set_bbox_to_anchor((1.4, 1))
            ax = plt.gca()
            ax.set(ylabel=None)
            # ax.legend_ = None
            plt.savefig("./save/gn_density_plots/density_plot_" + patient_name_list[j] + '_{}'.format(mode) + ".png", dpi=300, bbox_inches='tight')
            plt.clf()


if __name__ == '__main__':
    ##################################
    # Creating the classifier model
    ##################################


    # selected models, in order of validation folds (i.e. 0 -> fold 0)
    top_model_morphset_list = ['./save/top_gn_f0.ckpt',
                               './save/top_gn_f1.ckpt',
                               './save/top_gn_f2.ckpt',
                               './save/top_gn_f3.ckpt',
                               './save/top_gn_f4.ckpt']

    # Generating MorphSet results
    enc = EfficientNet.from_pretrained(options.encoder, include_top=True)
    a = list(enc.children())[0]
    b = list(enc.children())[1]
    c = list(enc.children())[2]
    cx = nn.Sequential(*list(c.children()))
    d = list(enc.children())[3]
    e = list(enc.children())[4]
    enc = nn.Sequential(*[a, b, cx, d, e])
    net = MorphSet(options.img_c, options.num_classes, enc, mode='setformer')

    if options.cuda:
        cudnn.benchmark = True
        net.cuda()
        net = nn.DataParallel(net)

    if options.num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    morph_preds = []
    morph_gts = []
    net.eval()
    for j in range(len(top_model_morphset_list)):
        checkpoint = torch.load(top_model_morphset_list[j])
        state_dict = checkpoint['state_dict']
        net.load_state_dict(state_dict)
        print("Loaded model", top_model_morphset_list[j])
        test_dataset = GN_Dataset(mode='test', input_size=(options.img_h, options.img_w), test=j)
        test_loader = DataLoader(test_dataset, batch_size=6,
                             shuffle=False, num_workers=options.num_workers, drop_last=False)
        test(mode='morphset')

