import argparse
import sys
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from dgl.data import GINDataset
from dataloader import NMDataLoader
import models
# from ginparser import Parser
from datetime import datetime

from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.metrics import precision_recall_curve, average_precision_score

from matching import utils
from matching.config import parse_encoder

USE_ORCA_FEATS = False # whether to use orca motif counts along with embeddings
MAX_MARGIN_SCORE = 1e9 # a very large margin score to given orca constraints

from collections import defaultdict

def train(args, net, trainloader, optimizer, logger, epoch):
    net.train()

    if args.method_type == "order":
        clf_opt = optim.Adam(net.clf_model.parameters(), lr=args.lr)

    running_loss = 0
    # setup the offset to avoid the overlap with mouse cursor
    bar = tqdm(range(args.epochs), unit='batch', position=2, file=sys.stdout)

    for pos, (pos_a, pos_b, neg_a, neg_b) in zip(bar, trainloader):

        # batch graphs will be shipped to device in forward part of model
        pos_a, pos_b, neg_a, neg_b = pos_a.to(args.device), pos_b.to(args.device), neg_a.to(args.device), neg_b.to(
            args.device)
        emb_pos_a, emb_pos_b = net.emb_model(pos_a), net.emb_model(pos_b)
        emb_neg_a, emb_neg_b = net.emb_model(neg_a), net.emb_model(neg_b)

        emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
        emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)
        labels = torch.tensor([1] + [0]).to(args.device)
        intersect_embs = None
        pred = net(emb_as, emb_bs)
        loss = net.criterion(pred, intersect_embs, labels)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        optimizer.step()

        if args.method_type == "order":
            with torch.no_grad():
                pred = net.predict(pred)
            net.clf_model.zero_grad()
            pred = net.clf_model(pred.unsqueeze(1))
            criterion = nn.NLLLoss()
            clf_loss = criterion(pred, labels)
            clf_loss.backward()
            clf_opt.step()
        pred = pred.argmax(dim=-1)
        acc = torch.mean((pred == labels).type(torch.float))
        train_loss = loss.item()
        train_acc = acc.item()

        # report
        bar.set_description('epoch-{}--Loss: {:.4f}. Training acc: {:.4f}'.format(epoch,train_loss, train_acc))
        logger.add_scalar("Loss/train", train_loss, epoch)
        logger.add_scalar("Accuracy/train", train_acc, epoch)

    bar.close()
    # the final batch will be aligned
    running_loss = running_loss / args.epochs

    return running_loss


def eval_net(args, net, dataloader,logger,batch_n,epoch, verbose=False):

    test_pts = []
    for pos_a, pos_b, neg_a, neg_b in dataloader:
        if pos_a:
            pos_a = pos_a.to(torch.device("cpu"))
            pos_b = pos_b.to(torch.device("cpu"))
        neg_a = neg_a.to(torch.device("cpu"))
        neg_b = neg_b.to(torch.device("cpu"))
        test_pts.append((pos_a, pos_b, neg_a, neg_b))

    net.eval()
    all_raw_preds, all_preds, all_labels = [], [], []
    for pos_a, pos_b, neg_a, neg_b in test_pts:
        if pos_a:
            pos_a = pos_a.to(args.device)
            pos_b = pos_b.to(args.device)
        neg_a = neg_a.to(args.device)
        neg_b = neg_b.to(args.device)
        labels = torch.tensor([1] +
            [0]).to(args.device)
        with torch.no_grad():
            emb_neg_a, emb_neg_b = (net.emb_model(neg_a),
                net.emb_model(neg_b))
            if pos_a:
                emb_pos_a, emb_pos_b = (net.emb_model(pos_a),
                    net.emb_model(pos_b))
                emb_as = torch.cat((emb_pos_a, emb_neg_a), dim=0)
                emb_bs = torch.cat((emb_pos_b, emb_neg_b), dim=0)
            else:
                emb_as, emb_bs = emb_neg_a, emb_neg_b
            pred = net(emb_as, emb_bs)
            raw_pred = net.predict(pred)
            if USE_ORCA_FEATS:
                import orca
                import matplotlib.pyplot as plt
                def make_feats(g):
                    counts5 = np.array(orca.orbit_counts("node", 5, g))
                    for v, n in zip(counts5, g.nodes):
                        if g.nodes[n]["node_feature"][0] > 0:
                            anchor_v = v
                            break
                    v5 = np.sum(counts5, axis=0)
                    return v5, anchor_v
                for i, (ga, gb) in enumerate(zip(neg_a.G, neg_b.G)):
                    (va, na), (vb, nb) = make_feats(ga), make_feats(gb)
                    if (va < vb).any() or (na < nb).any():
                        raw_pred[pos_a.num_graphs + i] = MAX_MARGIN_SCORE

            if args.method_type == "order":
                pred = net.clf_model(raw_pred.unsqueeze(1)).argmax(dim=-1)
                raw_pred *= -1
            elif args.method_type == "ensemble":
                pred = torch.stack([m.clf_model(
                    raw_pred.unsqueeze(1)).argmax(dim=-1) for m in net.models])
                for i in range(pred.shape[1]):
                    print(pred[:,i])
                pred = torch.min(pred, dim=0)[0]
                raw_pred *= -1
            elif args.method_type == "mlp":
                raw_pred = raw_pred[:,1]
                pred = pred.argmax(dim=-1)
        all_raw_preds.append(raw_pred)
        all_preds.append(pred)
        all_labels.append(labels)
    pred = torch.cat(all_preds, dim=-1)
    labels = torch.cat(all_labels, dim=-1)
    raw_pred = torch.cat(all_raw_preds, dim=-1)
    acc = torch.mean((pred == labels).type(torch.float))
    prec = (torch.sum(pred * labels).item() / torch.sum(pred).item() if
        torch.sum(pred) > 0 else float("NaN"))
    recall = (torch.sum(pred * labels).item() /
        torch.sum(labels).item() if torch.sum(labels) > 0 else
        float("NaN"))
    labels = labels.detach().cpu().numpy()
    raw_pred = raw_pred.detach().cpu().numpy()
    pred = pred.detach().cpu().numpy()
    auroc = roc_auc_score(labels, raw_pred)
    avg_prec = average_precision_score(labels, raw_pred)
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    if verbose:
        import matplotlib.pyplot as plt
        precs, recalls, threshs = precision_recall_curve(labels, raw_pred)
        plt.plot(recalls, precs)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.savefig("plots/precision-recall-curve.png")
        print("Saved PR curve plot in plots/precision-recall-curve.png")

    print("\n{}".format(str(datetime.now())))
    print("Validation. Epoch {}. Acc: {:.4f}. "
        "P: {:.4f}. R: {:.4f}. AUROC: {:.4f}. AP: {:.4f}.\n     "
        "TN: {}. FP: {}. FN: {}. TP: {}".format(epoch,
            acc, prec, recall, auroc, avg_prec,
            tn, fp, fn, tp))

    if not args.test:
        logger.add_scalar("Accuracy/test", acc, batch_n)
        logger.add_scalar("Precision/test", prec, batch_n)
        logger.add_scalar("Recall/test", recall, batch_n)
        logger.add_scalar("AUROC/test", auroc, batch_n)
        logger.add_scalar("AvgPrec/test", avg_prec, batch_n)
        logger.add_scalar("TP/test", tp, batch_n)
        logger.add_scalar("TN/test", tn, batch_n)
        logger.add_scalar("FP/test", fp, batch_n)
        logger.add_scalar("FN/test", fn, batch_n)
        print("Saving {}".format(args.model_path))
        torch.save(net.state_dict(), args.model_path)

    if verbose:
        conf_mat_examples = defaultdict(list)
        idx = 0
        for pos_a, pos_b, neg_a, neg_b in test_pts:
            if pos_a:
                pos_a = pos_a.to(args.device)
                pos_b = pos_b.to(args.device)
            neg_a = neg_a.to(args.device)
            neg_b = neg_b.to(args.device)
            for list_a, list_b in [(pos_a, pos_b), (neg_a, neg_b)]:
                if not list_a: continue
                for a, b in zip(list_a.G, list_b.G):
                    correct = pred[idx] == labels[idx]
                    conf_mat_examples[correct, pred[idx]].append((a, b))
                    idx += 1

def main(args):
    record_keys = ["conv_type", "n_layers", "hidden_dim","margin", "dataset", "max_graph_size", "skip"]
    args_str = ".".join(["{}={}".format(k, v)
        for k, v in sorted(vars(args).items()) if k in record_keys])
    logger = SummaryWriter(comment=args_str)
    # set up seeds, args.seed supported
    torch.manual_seed(seed=args.seed)
    np.random.seed(seed=args.seed)

    is_cuda = not args.disable_cuda and torch.cuda.is_available()
    if is_cuda:
        args.device = torch.device("cuda:" + str(args.device))
        torch.cuda.manual_seed_all(seed=args.seed)
    else:
        args.device = torch.device("cpu")

    dataset = GINDataset(args.dataset, self_loop = True, raw_dir="C:/Users/Administrator/.dgl", force_reload=True)

    trainloader, validloader = NMDataLoader(
        dataset, batch_size=args.batch_size, device=args.device,
        seed=args.seed, shuffle=True,
        split_name='fold10', fold_idx=args.fold_idx).train_valid_loader()

    model = models.OrderEmbedder(1, args.hidden_dim, args)

    criterion = nn.CrossEntropyLoss()  # defaul reduce is true
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    # it's not cost-effective to hanle the cursor and init 0
    # https://stackoverflow.com/a/23121189
    tbar = tqdm(range(args.epochs), unit="epoch", position=3, ncols=0, file=sys.stdout)
    vbar = tqdm(range(args.epochs), unit="epoch", position=4, ncols=0, file=sys.stdout)
    lrbar = tqdm(range(args.epochs), unit="epoch", position=5, ncols=0, file=sys.stdout)

    for epoch, _, _ in zip(tbar, vbar, lrbar):

        train(args, model, trainloader, optimizer, logger, epoch)
        scheduler.step()
# def eval_net(args, net, dataloader,logger,batch_n,epoch, verbose=False)

        eval_net(args, model, validloader,logger,0,0, verbose=False)

        if not args.filename == "":
            with open(args.filename, 'a') as f:
                f.write('%s %s %s %s' % (
                    args.dataset,
                    args.learn_eps,
                    args.neighbor_pooling_type,
                    args.graph_pooling_type
                ))
                f.write("\n")
                f.write("\n")

    tbar.close()
    vbar.close()
    lrbar.close()


if __name__ == '__main__':
    mp.set_start_method("spawn", force=True)
    parser = argparse.ArgumentParser(description='Order embedding arguments')
    utils.parse_optimizer(parser)
    parse_encoder(parser)
    args = parser.parse_args()

    print(args)
    main(args)
