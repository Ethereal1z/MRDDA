import os
import glob
import numpy as np
import pandas as pd
import torch as th
from warnings import simplefilter
from model import Model
from sklearn.model_selection import KFold
from load_data import load, remove_graph
from utils import get_metrics_auc, set_seed, m2v, plot_result_auc, \
    plot_result_aupr, EarlyStopping, get_metrics
from args import args
import scipy.io as sio


def train():
    simplefilter(action='ignore', category=FutureWarning)
    print(args)
    try:
        os.mkdir(args.saved_path)
    except:
        pass

    if args.device_id:
        print('Training on GPU')
        device = th.device('cuda:{}'.format(args.device_id))
    else:
        print('Training on CPU')
        device = th.device('cpu')

    # load DDA data for Kfold splitting
    if args.dataset in ['Kdataset', 'Bdataset']:
        df = pd.read_csv('./dataset/{}/{}_baseline.csv'.format(args.dataset, args.dataset), header=None).values
    else:
        m = sio.loadmat('./dataset/Cdataset/Cdataset.mat')
        df = m['didr'].T
    data = np.array([[i, j, df[i, j]] for i in range(df.shape[0]) for j in range(df.shape[1])])
    data = data.astype('int64')
    data_pos = data[np.where(data[:, -1] == 1)[0]]
    data_neg = data[np.where(data[:, -1] == 0)[0]]
    assert len(data) == len(data_pos) + len(data_neg)

    set_seed(args.seed)
    kf = KFold(n_splits=args.nfold, shuffle=True, random_state=args.seed)
    fold = 0
    for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kf.split(data_pos),
                                                                            kf.split(data_neg)):
        fold += 1
        print('{}-Cross Validation: Fold {}'.format(args.nfold, fold))

        # get the index list for train and test set
        train_pos_id, test_pos_id = data_pos[train_pos_idx], data_pos[test_pos_idx]
        train_neg_id, test_neg_id = data_neg[train_neg_idx], data_neg[test_neg_idx]
        train_pos_idx = [tuple(train_pos_id[:, 0]), tuple(train_pos_id[:, 1])]
        test_pos_idx = [tuple(test_pos_id[:, 0]), tuple(test_pos_id[:, 1])]
        train_neg_idx = [tuple(train_neg_id[:, 0]), tuple(train_neg_id[:, 1])]
        test_neg_idx = [tuple(test_neg_id[:, 0]), tuple(test_neg_id[:, 1])]
        assert len(test_pos_idx[0]) + len(test_neg_idx[0]) + len(train_pos_idx[0]) + len(train_neg_idx[0]) == len(data)

        g = load(args.dataset)
        print(g)
        # remove test set DDA from train graph
        g = remove_graph(g, test_pos_id[:, :-1]).to(device)
        if args.dataset == 'Kdataset':
            feature = {'drug': g.nodes['drug'].data['h'],
                       'disease': g.nodes['disease'].data['h'],
                       'protein': g.nodes['protein'].data['h'],
                       'gene': g.nodes['gene'].data['h'],
                       'pathway': g.nodes['pathway'].data['h']}
            metapath = ['disease_drug', 'drug_protein', 'protein_drug', 'drug_disease']
        elif args.dataset == 'Bdataset':
            feature = {'drug': g.nodes['drug'].data['h'],
                       'disease': g.nodes['disease'].data['h'],
                       'protein': g.nodes['protein'].data['h']}
            metapath = ['drug_protein', 'protein_drug', 'drug_disease']
        else:
            feature = {'drug': g.nodes['drug'].data['h'],
                       'disease': g.nodes['disease'].data['h']}
            metapath = ['drug_disease', 'disease_drug']

        # get the mask list for train and test set that used for performance calculation
        mask_label = np.ones(df.shape)
        mask_label[test_pos_idx[0], test_pos_idx[1]] = 0
        mask_label[test_neg_idx[0], test_neg_idx[1]] = 0
        mask_test = np.where(mask_label == 0)
        mask_test = [tuple(mask_test[0]), tuple(mask_test[1])]
        mask_train = np.where(mask_label == 1)
        mask_train = [tuple(mask_train[0]), tuple(mask_train[1])]

        print('Number of total training samples: {}, pos samples: {}, neg samples: {}'.format(len(mask_train[0]),
                                                                                              len(train_pos_idx[0]),
                                                                                              len(train_neg_idx[0])))
        print('Number of total testing samples: {}, pos samples: {}, neg samples: {}'.format(len(mask_test[0]),
                                                                                             len(test_pos_idx[0]),
                                                                                             len(test_neg_idx[0])))
        label = th.tensor(df).float().to(device)

        # load model and optimizer
        drug_emb, disease_emb = m2v(g, metapath)
        model = Model(etypes=g.etypes, ntypes=g.ntypes,
                      in_feats=feature['drug'].shape[1],
                      hidden_feats=args.hidden_feats,
                      num_heads=args.num_heads,
                      dropout=args.dropout)
        model.to(device)

        optimizer = th.optim.Adam(model.parameters(),
                                  lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
        optim_scheduler = th.optim.lr_scheduler.CyclicLR(optimizer,
                                                         base_lr=0.1 * args.learning_rate,
                                                         max_lr=args.learning_rate,
                                                         gamma=0.995,
                                                         step_size_up=20,
                                                         mode="exp_range",
                                                         cycle_momentum=False)
        criterion = th.nn.BCEWithLogitsLoss(pos_weight=th.tensor(len(train_neg_idx[0]) / len(train_pos_idx[0])))
        print('Loss pos weight: {:.3f}'.format(len(train_neg_idx[0]) / len(train_pos_idx[0])))
        stopper = EarlyStopping(patience=args.patience, saved_path=args.saved_path)

        # model training
        for epoch in range(1, args.epoch + 1):
            model.train()
            score = model(g, feature, drug_emb, disease_emb)
            pred = th.sigmoid(score)
            loss = criterion(score[mask_train].cpu().flatten(),
                             label[mask_train].cpu().flatten())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optim_scheduler.step()
            model.eval()
            AUC, _ = get_metrics_auc(label[mask_train].cpu().detach().numpy(),
                                      pred[mask_train].cpu().detach().numpy())
            early_stop = stopper.step(loss.item(), AUC, model)

            if epoch % 100 == 0:
                print('Epoch {} Loss: {:.3f}; Train AUC {:.3f}'.format(epoch, loss.item(), AUC))
                print('-' * 40)
                if early_stop:
                    break
    fold = 0
    dir = glob.glob(args.saved_path + '/*.pth')
    pred_result = np.zeros(df.shape)
    print('model testing')
    for (train_pos_idx, test_pos_idx), (train_neg_idx, test_neg_idx) in zip(kf.split(data_pos),
                                                                            kf.split(data_neg)):
        # get the index list for test set
        train_pos_id, test_pos_id = data_pos[train_pos_idx], data_pos[test_pos_idx]
        train_neg_id, test_neg_id = data_neg[train_neg_idx], data_neg[test_neg_idx]
        test_pos_idx = [tuple(test_pos_id[:, 0]), tuple(test_pos_id[:, 1])]
        test_neg_idx = [tuple(test_neg_id[:, 0]), tuple(test_neg_id[:, 1])]

        g = load(args.dataset)
        g = remove_graph(g, test_pos_id[:, :-1]).to(device)
        if args.dataset == 'Kdataset':
            feature = {'drug': g.nodes['drug'].data['h'],
                       'disease': g.nodes['disease'].data['h'],
                       'protein': g.nodes['protein'].data['h'],
                       'gene': g.nodes['gene'].data['h'],
                       'pathway': g.nodes['pathway'].data['h']}
            metapath = ['disease_drug', 'drug_protein', 'protein_drug', 'drug_disease']
        elif args.dataset == 'Bdataset':
            feature = {'drug': g.nodes['drug'].data['h'],
                       'disease': g.nodes['disease'].data['h'],
                       'protein': g.nodes['protein'].data['h']}
            metapath = ['drug_protein', 'protein_drug', 'drug_disease']
        else:
            feature = {'drug': g.nodes['drug'].data['h'],
                       'disease': g.nodes['disease'].data['h']}
            metapath = ['drug_disease', 'disease_drug']

        # get the mask list for test set that used for performance calculation
        mask_label = np.ones(df.shape)
        mask_label[test_pos_idx[0], test_pos_idx[1]] = 0
        mask_label[test_neg_idx[0], test_neg_idx[1]] = 0
        mask_test = np.where(mask_label == 0)
        mask_test = [tuple(mask_test[0]), tuple(mask_test[1])]

        assert len(mask_test[0]) == len(test_neg_idx[0]) + len(test_pos_idx[0])
        label = th.tensor(df).float().to(device)

        # load model and optimizer
        drug_emb, disease_emb = m2v(g, metapath)
        model = Model(etypes=g.etypes, ntypes=g.ntypes,
                      in_feats=feature['drug'].shape[1],
                      hidden_feats=args.hidden_feats,
                      num_heads=args.num_heads,
                      dropout=args.dropout)
        model.to(device)
        model.load_state_dict(th.load(dir[fold]))
        model.eval()
        pred = th.sigmoid(model(g, feature, drug_emb, disease_emb))
        AUC, AUPR = get_metrics_auc(label[mask_test].cpu().detach().numpy(), pred[mask_test].cpu().detach().numpy())
        pred = pred.cpu().detach().numpy()
        pred_result[test_pos_idx[0], test_pos_idx[1]] = pred[test_pos_idx[0], test_pos_idx[1]]
        pred_result[test_neg_idx[0], test_neg_idx[1]] = pred[test_neg_idx[0], test_neg_idx[1]]
        fold += 1
        print('Fold {} Test AUC {:.3f}; AUPR: {:.3f}'.format(fold, AUC, AUPR))

    # save the result
    AUC, aupr, acc, f1, pre, rec, spe = get_metrics(label.cpu().detach().numpy().flatten(), pred_result.flatten())
    print(
        'Overall: AUC {:.3f}; AUPR: {:.3f}; Acc: {:.3f}; F1: {:.3f}; Precision {:.3f}; Recall {:.3f}; Specificity {:.3F}'.
            format(AUC, aupr, acc, f1, pre, rec, spe))
    pd.DataFrame(pred_result).to_csv(os.path.join(args.saved_path, 'result.csv'), index=False, header=False)
    plot_result_auc(args, data[:, -1].flatten(), pred_result.flatten(), AUC)
    plot_result_aupr(args, data[:, -1].flatten(), pred_result.flatten(), aupr)


if __name__ == '__main__':
    train()
