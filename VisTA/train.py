import os

import numpy as np
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.distributed as dist

from utils.logger import get_logger

import argparse

from dataset.DataNew import DatasetNew
from model.VisTA import VisTA

# device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True

batch_size = 16
workers = 8
lr=1e-4
max_epoch=50

TITLE = 'VisTA'

ckp_savepath = './ckps/' + TITLE
log_savepath = './logs/'
if not os.path.exists(ckp_savepath):
    os.makedirs(ckp_savepath)
if not os.path.exists(log_savepath):
    os.makedirs(log_savepath)
logger = get_logger(log_savepath + TITLE + '.log')
scaler = torch.GradScaler()


def confuse_matrix(pre, label, n_class):
    pre = pre.cpu().numpy()
    label = label.cpu().numpy()
    cm = np.bincount(label * n_class + pre, minlength=n_class ** 2).reshape(n_class, n_class)
    return cm


def ddp_setup():
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12751"
    init_process_group(backend="nccl")
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))


def main(args):
    ddp_setup()

    train_root = './train/'
    val_root='./val/'


    train_dataset = DatasetNew(train_root+'json/Train.json', train_root+'image/', train_root+'mask/')
    train_sampler = DistributedSampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True,
                              sampler=train_sampler)

    val_dataset = DatasetNew(val_root+'json/Val.json', val_root+'image/', val_root+'mask/')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True,
                            sampler=DistributedSampler(val_dataset))

    criterion = nn.CrossEntropyLoss()

    gpu_id = int(os.environ['LOCAL_RANK'])

    model=VisTA()

    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model.to(gpu_id), device_ids=[gpu_id], find_unused_parameters=True)


    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.)
    scheduler = MultiStepLR(optimizer, milestones=[35], gamma=0.1)
    best_acc = 0
    best_epoch = 0
    for epoch in range(0, max_epoch):
        train_sampler.set_epoch(epoch + 1)

        train_loss, train_acc, train_f1, train_iou, train_loss1, train_loss2 = train(train_loader=train_loader,
                                                                                     criterion=criterion, model=model,
                                                                                     optimizer=optimizer,
                                                                                     gpu_id=gpu_id, )
        if gpu_id == 0:
            logger.info(
                'Epoch:[{}/{}]\t train_loss={:.4f}\t train_ACC={:.8f}\t train_F1={:.8f}\t train_IoU={:.8f}\t loss1={:.4f}\t loss2={:.4f}\t'.format(
                    epoch, max_epoch,
                    train_loss, train_acc,
                    train_f1, train_iou, train_loss1, train_loss2))

        val_acc, _, f1, iou = validate(val_loader=val_loader, model=model, gpu_id=gpu_id)
        if val_acc > best_acc and gpu_id == 0:
            best_acc = val_acc
            best_epoch = epoch
            ckp_name = 'epoch:{}_acc:{:.4f}_f1:{:.4f}.pth'.format(epoch, val_acc, f1)
            torch.save(model.module.state_dict(), os.path.join(ckp_savepath, TITLE + ckp_name))
        if gpu_id == 0:
            logger.info(
                'Epoch:[{}/{}]\t val_ACC={:.8f}\t  F1={:.8f}\t IoU={:.8f}\t best_epoch={}\t best_ACC={:.4f}\t'.format(
                    epoch,
                    max_epoch,
                    val_acc, f1, iou,
                    best_epoch,
                    best_acc))
        scheduler.step()


def train(train_loader, criterion, gpu_id, model, optimizer):
    model.train()

    epoch_loss = 0
    loss1_sum = 0
    loss2_sum = 0
    cm_all = np.zeros((23, 23))
    tp = 0
    fp = 0
    fn = 0

    for imgs1, imgs2, q_str, type_str, answer_vec, mask_img, _, in tqdm(train_loader):
        imgs1 = imgs1.to(gpu_id).bfloat16()
        imgs2 = imgs2.to(gpu_id).bfloat16()
        answer_vec = answer_vec.to(gpu_id).bfloat16()
        mask_img = mask_img.to(gpu_id).bfloat16()

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type='cuda', enabled=True,dtype=torch.bfloat16):
            pred, ans, target, loss, loss1, loss2 = model(imgs1, imgs2, q_str, mask_img, answer_vec)

        pred = pred.flatten(1)
        target = target.flatten(1)
        pred = torch.sigmoid(pred)
        pred[pred < 0.35] = 0.
        pred[pred >= 0.35] = 1.

        pred = pred.cpu().float().numpy()
        target = target.cpu().float().numpy()

        tp += np.sum((pred == 1) & (target == 1))
        fp += np.sum((pred == 1) & (target == 0))
        fn += np.sum((pred == 0) & (target == 1))

        ans = ans.argmax(dim=1)
        answer_vec = answer_vec.argmax(dim=1)
        cm_all += confuse_matrix(ans, answer_vec, 23)

        epoch_loss += loss.item()
        loss1_sum += loss1.item()
        loss2_sum += loss2.item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    cm_all = torch.from_numpy(cm_all).to(gpu_id)
    tp = torch.tensor(tp).to(gpu_id)
    fp = torch.tensor(fp).to(gpu_id)
    fn = torch.tensor(fn).to(gpu_id)

    dist.all_reduce(cm_all)
    dist.all_reduce(tp)
    dist.all_reduce(fp)
    dist.all_reduce(fn)

    cm_all = cm_all.cpu().numpy()
    tp = tp.cpu().numpy()
    fp = fp.cpu().numpy()
    fn = fn.cpu().numpy()

    return (epoch_loss, cm_all.diagonal().sum() / cm_all.sum(), 2 * tp / (2 * tp + fp + fn),
            tp / (tp + fp + fn), loss1_sum, loss2_sum)


def validate(val_loader, gpu_id, model):
    cm_all = np.zeros((23, 23))
    model.eval()
    tp = 0
    fp = 0
    fn = 0

    for imgs1, imgs2, q_str, type_str, answer_vec, mask_img, _,in val_loader:
        imgs1 = imgs1.to(gpu_id)
        imgs2 = imgs2.to(gpu_id)
        answer_vec = answer_vec.to(gpu_id)
        mask_img = mask_img.to(gpu_id)

        with torch.no_grad():
            pred, ans = model(imgs1, imgs2, q_str, mask_img)

        pred = pred.flatten(1)
        target = mask_img.flatten(1)
        pred[pred < 0.35] = 0.
        pred[pred >= 0.35] = 1.

        pred = pred.cpu().float().numpy()
        target = target.cpu().float().numpy()

        tp += np.sum((pred == 1) & (target == 1))
        fp += np.sum((pred == 1) & (target == 0))
        fn += np.sum((pred == 0) & (target == 1))

        ans = ans.argmax(dim=1)
        answer_vec = answer_vec.argmax(dim=1)
        cm_all += confuse_matrix(ans, answer_vec, 23)

    cm_all = torch.from_numpy(cm_all).to(gpu_id)
    tp = torch.tensor(tp).to(gpu_id)
    fp = torch.tensor(fp).to(gpu_id)
    fn = torch.tensor(fn).to(gpu_id)

    dist.all_reduce(cm_all)
    dist.all_reduce(tp)
    dist.all_reduce(fp)
    dist.all_reduce(fn)

    cm_all = cm_all.cpu().numpy()
    tp = tp.cpu().numpy()
    fp = fp.cpu().numpy()
    fn = fn.cpu().numpy()

    return cm_all.diagonal().sum() / cm_all.sum(), cm_all, 2 * tp / (2 * tp + fp + fn), tp / (tp + fp + fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
