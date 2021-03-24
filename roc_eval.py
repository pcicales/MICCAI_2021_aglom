import torch
import torch.nn as nn
import pandas as pd
import torch.backends.cudnn as cudnn
from ABMR_dataloader import ABMR_Dataset
from AMRDataset import AMRDataset
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
from models import *
from config import options
import numpy as np
import seaborn as sns
from utils.eval_utils import compute_accuracy
from sklearn import metrics
import matplotlib.pyplot as plt
from utils.logger_utils import Logger
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
    with torch.no_grad():
        if (mode == 'morphset') or (mode =='convbase'):
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
            return(out_mean.tolist(), targets.tolist())
        else:
            test_loss = 0
            targets, outputs = [], []
            for batch_id, (data, target) in enumerate(test_loader):
                if options.cuda:
                    data, target = data.cuda(), target.cuda()
                output = bnet(data)
                if options.num_classes == 2:
                    batch_loss = criterion(output, target.type_as(output).unsqueeze(1))
                else:
                    batch_loss = criterion(output, target)
                targets += [target]
                outputs += [torch.sigmoid(output)]
                test_loss += batch_loss

            test_loss /= (batch_id + 1)
            test_acc = compute_accuracy(torch.cat(targets), torch.cat(outputs), options.num_classes)
            print('Validation set accuracy for fold {}: {}'.format(j, test_acc))
            print('Validation set loss for fold {}: {}'.format(j, test_loss))
            sample_name_list = list(test_loader.dataset.images)
            return(torch.cat(outputs).tolist(), torch.cat(targets).tolist(), sample_name_list)



if __name__ == '__main__':

    #######################################
    # Use cuda and load in pretrained model
    #######################################
    # selected models, in order of validation folds (i.e. 0 -> fold 0)
    top_model_morphset_list = ['./save/top_f0.ckpt',
                               './save/top_f1.ckpt',
                               './save/top_f2.ckpt',
                               './save/top_f3.ckpt',
                               './save/top_f4.ckpt']

    top_model_baseline_list = ['./save/top_fine_f0.ckpt',
                               './save/top_fine_f1.ckpt',
                               './save/top_fine_f2.ckpt',
                               './save/top_fine_f3.ckpt',
                               './save/top_fine_f4.ckpt']

    top_model_conv_list = ['./save/top_ab_f0.ckpt',
                           './save/top_ab_f1.ckpt',
                           './save/top_ab_f2.ckpt',
                           './save/top_ab_f3.ckpt',
                           './save/top_ab_f4.ckpt']

    # Generating EfficientNet results
    if options.num_classes == 2:
        outputs = 1
    else:
        outputs = options.num_classes
    bnet = EfficientNet.from_pretrained('efficientnet-b3', include_top=True, num_classes=outputs)

    if options.cuda:
        cudnn.benchmark = True
        bnet.cuda()
        bnet = nn.DataParallel(bnet)

    if options.num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    fine_preds = []
    fine_gts = []
    fine_names = []
    fine_patients = {}
    bnet.eval()
    for j in range(len(top_model_baseline_list)):
        checkpoint = torch.load(top_model_baseline_list[j])
        state_dict = checkpoint['state_dict']
        bnet.load_state_dict(state_dict)
        print("Loaded model", top_model_baseline_list[j])
        test_dataset = AMRDataset(mode='test', test=j)
        for file in test_dataset.images.tolist():
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
            if patient_name in fine_patients:
                fine_patients[patient_name].append(file)
            else:
                fine_patients[patient_name] = [file]
        test_loader = DataLoader(test_dataset, batch_size=3*options.batch_size,
                                 shuffle=False, num_workers=options.num_workers, drop_last=False)
        fpreds, ftargets, fnames = test(mode='finegrained')
        fine_preds += fpreds
        fine_gts += ftargets
        fine_names += fnames

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
        test_dataset = ABMR_Dataset(mode='test', test=j)
        test_loader = DataLoader(test_dataset, batch_size=3*options.batch_size,
                                 shuffle=False, num_workers=options.num_workers, drop_last=False)
        mpreds, mgts = test(mode='morphset')
        morph_preds += mpreds
        morph_gts += mgts

    morph_preds = [item for sublist in morph_preds for item in sublist]

    # Generating ConvSet results
    enc = EfficientNet.from_pretrained(options.encoder, include_top=True)
    a = list(enc.children())[0]
    b = list(enc.children())[1]
    c = list(enc.children())[2]
    cx = nn.Sequential(*list(c.children()))
    d = list(enc.children())[3]
    e = list(enc.children())[4]
    enc = nn.Sequential(*[a, b, cx, d, e])
    cnet = MorphSet(options.img_c, options.num_classes, enc, mode='conv')

    if options.cuda:
        cudnn.benchmark = True
        cnet.cuda()
        cnet = nn.DataParallel(cnet)

    if options.num_classes == 2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    conv_preds = []
    conv_gts = []
    cnet.eval()
    for j in range(len(top_model_conv_list)):
        checkpoint = torch.load(top_model_conv_list[j])
        state_dict = checkpoint['state_dict']
        cnet.load_state_dict(state_dict)
        print("Loaded model", top_model_conv_list[j])
        test_dataset = ABMR_Dataset(mode='test', test=j)
        test_loader = DataLoader(test_dataset, batch_size=3*options.batch_size,
                                 shuffle=False, num_workers=options.num_workers, drop_last=False)
        cpreds, cgts = test(mode='convbase')
        conv_preds += cpreds
        conv_gts += cgts

    conv_preds = [item for sublist in conv_preds for item in sublist]

    # Generating the patient performance vector for fine grained
    flat_preds = [item for sublist in fine_preds for item in sublist]
    bool_preds = np.array(flat_preds) > 0.5
    final_preds = bool_preds.astype(int)

    perc_list = []
    gt_list = []
    for patient in fine_patients:
        amr_total = 0
        total = len(fine_patients[patient])  # patients image list length in dictionary
        ground_truth = 1 if patient[0] == 'A' else 0
        for image_filename in fine_patients[patient]:
            idx = fine_names.index(image_filename)  # Find index of image filename in list
            # Use index to find its corresponding prediction, add it
            amr_total += flat_preds[idx]
        perc = amr_total / total
        perc_list += [perc]
        gt_list += [ground_truth]

    # Efficientnet
    fpr, tpr, thresholds = metrics.roc_curve(np.array(gt_list),
                                             np.array(perc_list),
                                             pos_label=1)  # If two output neurons, do torch.argmax() on outputs
    roc_auc = metrics.auc(fpr, tpr)

    # MorphSet
    fprm, tprm, thresholdsm = metrics.roc_curve(np.array(morph_gts),
                                             np.array(morph_preds),
                                             pos_label=1)  # If two output neurons, do torch.argmax() on outputs
    roc_aucm = metrics.auc(fprm, tprm)

    # Conv. Base
    fprc, tprc, thresholdsc = metrics.roc_curve(np.array(conv_gts),
                                                np.array(conv_preds),
                                                pos_label=1)  # If two output neurons, do torch.argmax() on outputs
    roc_aucc = metrics.auc(fprc, tprc)

    # Figures
    set_style()
    fig = plt.figure()
    plt.plot(fpr, tpr, label='EfficientNet-B3 (AUC = {0:0.3f})'.format(roc_auc))
    plt.plot(fprc, tprc, label='Conv. Baseline (AUC = {0:0.3f})'.format(roc_aucc))
    plt.plot(fprm, tprm, label='MorphSet (AUC = {0:0.3f})'.format(roc_aucm))
    plt.legend(loc='lower right')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    fig.savefig('ROC_curve.pdf')
    print('Done')

