import os

from torch.utils.data import TensorDataset, DataLoader
from dataset.load_dataset import load_dataset
from model.AttentionNetwork3D import AttentionNetwork3D
from model.DenseNet3D import densenet121
from model.I3D import I3D
from model.Resnext3D import resnext50_32x4d
from model.SSFTT import SSFTTnet
from model.context_cluster import coc_tiny
from model.efficientnet3d import EfficientNet
from model.gfnet_3d_3 import GFNet
from model.mobilenet3d import get_model
# from model.resnet3d import generate_model
from model.senet3d import se_resnet34

from model.shufflenetv2_3d import shufflenet_v2_x2_0
from model.swin_3d import SwinTransformer3D
from model.vit_3d import ViT

# from model.resnet3d import generate_model
# from model_multime.resnet3d_multime import generate_model
# from  model.resnet import resnet18
# from model_multime.resnet3d_multime_fusion import generate_model
# from model_multime.resnet3d_multime_try import generate_model
from model_multime.resnet3d_multime_pointwise import generate_model

from train.data_loader import get_loader

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics, manifold
from configs import load_config
from sklearn.model_selection import KFold  # �sklearn�eKFold
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.model_selection import train_test_split
import csv
import random
import scipy.io as scio

# path = 'checkpoint_model_best(tt1)_10fold-p_bs=8.pth'
count = 0
c1 = 'CN'
c2 = 'MCI'
# model_name = ['resnet', 'I3D', 'efficientnet', 'senet', 'att3d']
model_name = ['senet']
name = ['senet']
pretrain = False


def select_ad_cn_mci(path_csv, select):
    scores_list = []
    labels = []
    with open(path_csv) as csv_file:  # Phenototal1.csv
        reader = csv.DictReader(csv_file)
        for row in reader:
            if row['category'] in select[0]:
                scores_list.append(row['id'])
                labels.append(0)
            if row['category'] in select[1]:
                scores_list.append(row['id'])
                labels.append(1)
    num = len(labels)
    return scores_list, labels, num


def create_model(name, classes):
    global model
    if name == 'resnet':
        model = generate_model(model_depth=18, n_classes=2)
        # model = resnet18(pretrained=False, progress=True, num_classes=2)
    elif name == 'resnext':
        model = resnext50_32x4d(num_classes=classes)
    elif name == 'mobilenet':
        model = get_model(num_classes=classes, sample_size=120, width_mult=1.)
    elif name == 'shufflenet':
        model = shufflenet_v2_x2_0(num_classes=classes)
    elif name == 'densenet':
        model = densenet121()
    elif name == 'efficientnet':
        model = EfficientNet(in_channels=1, num_classes=classes)
    elif name == 'att3d':
        model = AttentionNetwork3D(num_classes=classes)
    elif name == 'context_cluster':
        model = coc_tiny(num_classes=classes)
    elif name == 'vit':
        model = ViT(image_size=(120, 100), image_patch_size=(10, 10), frames=100, frame_patch_size=10,
                    num_classes=classes, dim=128, depth=8, heads=16, mlp_dim=1024, channels=1)
    elif name == 'senet':
        model = se_resnet34(num_classes=2)
    elif name == 'gfnet':
        model = GFNet(img_size=(100, 120, 100), num_cmodulelasses=classes)
    elif name == 'I3D':
        model = I3D(num_class=2, in_chans=1)
    elif name == 'SSFTT':
        model = SSFTTnet()
    elif name == 'swin':
        model = SwinTransformer3D(patch_size=(4, 4, 4),
                                  in_chans=1,
                                  embed_dim=96,
                                  window_size=(2, 7, 7))

    return model


def save_checkpoint(best_acc, model, optimizer, args, epoch):
    count = 2
    path_sub = r'/home/ubuntu/guanzhenghua/work2/cam_ckpt/cam1/checkpoint_{2}_{3}_{0}_{1}.pth'.format(c1, c2, count, name)
    path_checkpoints = os.path.join(args.path_dir, path_sub)
    print(path_checkpoints)
    print('Best Model Saving...')
    if args.device_num > 1:
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()

    torch.save({
        'model_state_dict': model_state_dict,
        'global_epoch': epoch,
        'optimizer_state_dict': optimizer.state_dict(),
        'best_acc': best_acc,
    }, path_checkpoints)


# �epn�P(numpyp�(list<�e�
def K_Flod_spilt(K, fold, data, label):
    '''
    :param K: ��pn���p�A!A��K=10
    :param fold: ��,���pn���,5� flod=5
    :param data:  �W�pn
    :param label: ��� �W~
    :return: �������K�ƌ���~
    '''
    split_train_list = []
    split_test_list = []
    kf = KFold(n_splits=K)
    for train, test in kf.split(data):
        split_train_list.append(train.tolist())
        split_test_list.append(test.tolist())
    train, test = split_train_list[fold], split_test_list[fold]
    return data[train], data[test], label[train], label[test]  # ��}W�pn�


def t_sne(x, label, args):
    tsne = manifold.TSNE(n_components=2)  # dataset [N, dim]
    X_tsne = tsne.fit_transform(x)
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)
    plt.figure()
    True_labels = label.reshape((-1, 1))

    S_data = np.hstack((X_norm, True_labels))
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})

    # plt.rc('font', family='Times New Roman')
    colors = ['#8e6fad', '#3e9d35']
    l = ['{}'.format(c2), '{}'.format(c1)]
    for index in range(2):
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, s=3, c=colors[index])

    plt.legend(l, prop={'size': 12})
    plt.xticks([])  # ��*P<
    plt.yticks([])  # ���P<

    path_sub = r't_sne/{2}_{3}_{0}_{1}_01_lstm.jpg'.format(c1, c2, count, name)
    path_tsne = os.path.join(args.path_dir, path_sub)
    plt.savefig(path_tsne, dpi=500)


def _train(epoch, train_loader, model, optimizer, criterion_cls, args):
    model.train()

    losses = 0.
    acc = 0.
    total = 0.
    for idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.long().cuda()
        output = model(data)
        _, pred = F.softmax(output, dim=-1).max(1)
        acc += pred.eq(target).sum().item()
        total += target.size(0)

        loss = criterion_cls(output, target)
        losses += loss
        loss.backward()

        if args.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2)
        optimizer.step()
        # if idx % 40 == 0 and idx != 0:
    print('Train--[{0}][Epoch: {1:4d}], Loss: {2:.3f},  Acc: {3:.3f}, Correct {4} / Total {5}'.format(count, epoch,
                                                                                                      losses / (
                                                                                                              idx + 1),
                                                                                                      acc / total * 100.,
                                                                                                      acc, total))
    return acc / total * 100.


def _eval(epoch, test_loader, model, args):
    model.eval()

    acc = 0.
    pred_matrix = []
    target_matrix = []
    TP = 0.
    FN = 0.
    FP = 0.
    TN = 0.
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            if args.cuda:
                data, target = data.cuda(), target.long().cuda()
            output = model(data)
            _, pred = F.softmax(output, dim=-1).max(1)
            for i in range(len(pred)):
                pred_matrix.append(pred[i].cpu())
                target_matrix.append(target[i].cpu())
            acc += pred.eq(target).sum().item()
            matrix = metrics.confusion_matrix(target_matrix, pred_matrix)
        TP += matrix[0, 0]
        FN += matrix[0, 1]
        FP += matrix[1, 0]
        TN += matrix[1, 1]
        # matrix = metrics.confusion_matrix(target_matrix, pred_matrix)
        precision = TP / (TP + FP + 1e-8)
        Sen = TP / (TP + FN + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        Spe = TN / (TN + FP + 1e-8)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        print(
            'Val--[{0}][Epoch: {1:4d}], Acc: {2:.3f}, Sen: {3:.4f}, Spe: {4:.4f}, F1: {5:.4f}, BAC: {6:.4f}'.format(
                count, epoch, acc / len(test_loader.dataset) * 100., Sen, Spe, f1_score, (Sen + Spe) / 2))

    return acc / len(test_loader.dataset) * 100., Sen, Spe, f1_score, (Sen + Spe) / 2


def _test(epoch, test_loader, model, args):
    model.eval()

    acc = 0.
    pred_matrix = []
    target_matrix = []
    # output_tsne = np.zeros(shape=[1, 2])
    output_tsne = []
    target_tsne = []
    score = np.zeros(shape=[1, 2])
    label1 = []
    TP = 0.
    FN = 0.
    FP = 0.
    TN = 0.
    with torch.no_grad():
        for idx, (data, target) in enumerate(test_loader):
            if args.cuda:
                data, target = data.cuda(), target.long().cuda()
            # output, att_out, roi_out = model(data)
            output = model(data)
            if idx == 0:
                output_tsne = output.cpu().numpy()
                target_tsne = target.cpu().numpy()
            else:
                output_tsne = np.vstack([output_tsne, output.cpu().numpy()])
                target_tsne = np.concatenate([target_tsne, target.cpu().numpy()], axis=0)
            # target_tsne.extend(target.cpu().numpy())
            # output_tsne = output.cpu().numpy()
            _, pred = F.softmax(output, dim=-1).max(1)
            out = F.softmax(output, dim=-1)
            if idx == 0:
                score[0][0] = out[0][0].cpu().numpy()
                score[0][1] = out[0][1].cpu().numpy()
            else:
                score = np.vstack([score, out.cpu().numpy()])
            # score.extend(out.cpu().numpy())
            for k in range(len(pred)):
                pred_matrix.append(pred[k].cpu())
                target_matrix.append(target[k].cpu())
            acc += pred.eq(target).sum().item()
            bio_onehot = np.empty(shape=[0, 2])
            # label = label_binarize(target_matrix, classes=np.array(list(range(2))))
            for i, value in enumerate(target_matrix):
                if value == 0:
                    bio_onehot = np.concatenate((bio_onehot, [[1, 0]]), 0)
                if value == 1:
                    bio_onehot = np.concatenate((bio_onehot, [[0, 1]]), 0)
            label = bio_onehot
            matrix = metrics.confusion_matrix(target_matrix, pred_matrix)
        TP += matrix[0, 0]
        FN += matrix[0, 1]
        FP += matrix[1, 0]
        TN += matrix[1, 1]

        # t_sne(output_tsne, label.ravel())
        # matrix = metrics.confusion_matrix(target_matrix, pred_matrix)
        precision = TP / (TP + FP + 1e-8)
        Sen = TP / (TP + FN + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        Spe = TN / (TN + FP)
        f1_score = 2 * precision * recall / (precision + recall + 1e-8)
        fpr, tpr, theresholds = metrics.roc_curve(label.ravel(), score.ravel(), pos_label=1, drop_intermediate=False)
        auc = metrics.auc(fpr, tpr)
        print(
            'Test--[Epoch: {0:4d}], Acc: {1:.3f}, Sen: {2:.4f}, Spe: {3:.4f}, F1: {4:.4f}, BAC: {5:.4f}, AUC: {6:.4f}'.format(
                epoch, acc / len(test_loader.dataset) * 100., Sen, Spe, f1_score, (Sen + Spe) / 2, auc))

    return acc / len(test_loader.dataset) * 100., Sen, Spe, f1_score, (
            Sen + Spe) / 2, auc, fpr, tpr, output_tsne, target_tsne, matrix, label.ravel(), score.ravel()


def main(args):
    global count, c1, c2, model_name, name, global_epoch, pretrain

    ad_cn_mci = [['AD', 'CN'], ['AD', 'MCI'], ['CN', 'MCI']]
    item_select = 0
    c1, c2 = ad_cn_mci[item_select][0], ad_cn_mci[item_select][1]
    select_item, labels, num = select_ad_cn_mci(args.path_csv, ad_cn_mci[item_select])
    data_with_labels = list(zip(select_item, labels))
    random.seed(19951012)
    random.shuffle(data_with_labels)
    select_item, labels = zip(*data_with_labels)
    print('classifier is : {}'.format(ad_cn_mci[item_select]))
    sfolder = StratifiedKFold(n_splits=10, random_state=19951012, shuffle=True)

    for name in model_name:
        test_acc = []
        test_sen = []
        test_spe = []
        test_bac = []
        test_f1 = []
        test_auc = []
        test_fpr = []
        test_tpr = []
        test_label = []
        test_scorn = []
        test_feature = []
        test_target = []
        test_matrix = []
        count = 0

        for train_ind, test_ind in sfolder.split(select_item, labels):
            count += 1
            best_val_acc = 0
            best_val_f1 = 0
            result_dict = {}
            train_list_id = []
            train_list_label = []
            test_list_id = []
            test_list_label = []

            for i in train_ind:
                train_list_id.append(select_item[i])
                train_list_label.append(labels[i])

            for i in test_ind:
                test_list_id.append(select_item[i])
                test_list_label.append(labels[i])



            # train_list_id, val_list_id, train_list_label, val_list_label = train_test_split(train_list_id,
            #                                                                                 train_list_label,
            #                                                                                 test_size=0.1,
            #                                                                                 stratify=train_list_label,
            #                                                                                 random_state=19951012)
            print('train {}:'.format(ad_cn_mci[item_select][0]), train_list_label.count(0), end=' ' * 5)
            print('train {}:'.format(ad_cn_mci[item_select][1]), train_list_label.count(1))
            # print('val {}:'.format(ad_cn_mci[item_select][0]), val_list_label.count(0), end=' ' * 5)
            # print('val {}:'.format(ad_cn_mci[item_select][1]), val_list_label.count(1))
            print('test {}:'.format(ad_cn_mci[item_select][0]), test_list_label.count(0), end=' ' * 5)
            print('test {}:'.format(ad_cn_mci[item_select][1]), test_list_label.count(1))
            print('-----------------------------------fold {}-----------------------------'.format(count))
            train = get_loader(
                data_path=args.path_all,
                list_id=train_list_id,
                list_label=train_list_label,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                mode='train',
                augmentation_prob=0.5,
            )

            # val = get_loader(
            #     data_path=args.path_all,
            #     list_id=val_list_id,
            #     list_label=val_list_label,
            #     batch_size=1,
            #     num_workers=args.num_workers,
            #     mode='valid',
            #     augmentation_prob=0.5,
            # )
            test = get_loader(
                data_path=args.path_all,
                list_id=test_list_id,
                list_label=test_list_label,
                batch_size=1,
                num_workers=args.num_workers,
                mode='test',
                augmentation_prob=0.5,
            )

            # train_data, test_data, label_train, label_test = K_Flod_spilt(k, ii, dataset, label)

            model = create_model(name=name, classes=2)
            optimizer = optim.AdamW(model.parameters(), lr=1e-5,  )

            start_epoch = 1

            if args.cuda:
                model = model.cuda()
            criterion_cls = nn.CrossEntropyLoss(label_smoothing=0.3)
            # lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=3, eta_min=1e-7)
            lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=3,)
            for epoch in range(start_epoch, args.epochs + 1):
                train_acc = _train(epoch, train, model, optimizer, criterion_cls, args)
                best_acc, best_sen, best_spe, best_f1_score, best_bac = _eval(
                    epoch, test, model, args)
                acc, sen, spe, f1_score, bac, auc, fpr, tpr, feature, target, matrix, label_roc, scorn_roc = _test(
                    epoch, test,
                    model, args)

                if best_val_acc < best_acc:
                    best_val_acc = best_acc
                    best_val_f1 = best_f1_score
                    print('-------------------above is the best val acc--------------------')
                    result_dict['acc'] = acc
                    result_dict['sen'] = sen
                    result_dict['spe'] = spe
                    result_dict['f1_score'] = f1_score
                    result_dict['bac'] = bac
                    result_dict['auc'] = auc
                    result_dict['fpr'] = fpr
                    result_dict['tpr'] = tpr
                    result_dict['feature'] = feature
                    result_dict['target'] = target
                    result_dict['matrix'] = matrix
                    save_checkpoint(best_acc, model, optimizer, args, epoch)
                elif best_val_acc == best_acc and best_val_f1 < best_f1_score:
                    best_val_acc = best_acc
                    best_val_f1 = best_f1_score
                    print('-------------------above is the best val acc--------------------')
                    result_dict['acc'] = acc
                    result_dict['sen'] = sen
                    result_dict['spe'] = spe
                    result_dict['f1_score'] = f1_score
                    result_dict['bac'] = bac
                    result_dict['auc'] = auc
                    result_dict['fpr'] = fpr
                    result_dict['tpr'] = tpr
                    result_dict['feature'] = feature
                    result_dict['target'] = target
                    result_dict['matrix'] = matrix
                    save_checkpoint(best_acc, model, optimizer, args, epoch)
                lr_scheduler.step()
                print('Current Learning Rate: {}'.format(lr_scheduler.get_last_lr()))

            print('Acc: {0:.3f}, Sen: {1:.4f}, Spe: {2:.4f}, F1: {3:.4f}, BAC: {4:.4f}, AUC: {5:.4f}'.format(
                result_dict['acc'], result_dict['sen'], result_dict['spe'], result_dict['f1_score'],
                result_dict['bac'], result_dict['auc'], ))
            test_acc.append(result_dict['acc'])
            test_sen.append(result_dict['sen'])
            test_spe.append(result_dict['spe'])
            test_f1.append(result_dict['f1_score'])
            test_bac.append(result_dict['bac'])
            test_auc.append(result_dict['auc'])
            test_label.append(label_roc)
            test_scorn.append(scorn_roc)
            test_fpr.append(fpr)
            test_tpr.append(tpr)
            if count == 1:
                test_feature = feature
                test_target = target
                test_matrix = matrix
            else:
                test_feature = np.vstack([test_feature, feature])
                test_target = np.concatenate([test_target, target], axis=0)
                test_matrix = test_matrix + matrix

            break
        test_fpr = np.array(test_fpr, dtype=object)
        test_tpr = np.array(test_tpr, dtype=object)
        test_label = np.array(test_label, dtype=object)
        test_scorn = np.array(test_scorn, dtype=object)
        result_cls = {'ACC': test_acc, 'SEN': test_sen, 'SPE': test_spe, 'F1': test_f1, 'BAC': test_bac,
                      'AUC': test_auc}
        path_fprtpr = os.path.join(args.path_dir, r'roc/fpr_tpr_lasbel_scorn.mat')

        path_feature = os.path.join(args.path_dir, r'test_feature/test_feature.npy')
        path_target = os.path.join(args.path_dir, r'test_target/test_target.npy')
        path_text = os.path.join(args.path_dir, r'text/text.npy')

        np.save(path_text, result_cls)
        # scio.savemat(path_fprtpr,  {'fpr': test_fpr, 'tpr': test_tpr,'label':test_label,'scorn':test_scorn})
        scio.savemat(path_fprtpr, {'fpr': test_fpr, 'tpr': test_tpr, 'label': test_label, 'scorn': test_scorn})

        np.save(path_feature, test_feature)
        np.save(path_target, test_target)
        t_sne(test_feature, test_target, args)


if __name__ == '__main__':
    args = load_config()
    main(args)
