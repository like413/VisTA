import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.DataNew import DatasetNew
from model.VisTA import VisTA

w2v_dict = {'0': 0, '0_to_10': 1, '10_to_20': 2, '20_to_30': 3, '30_to_40': 4, '40_to_50': 5, '50_to_60': 6,
            '60_to_70': 7, '70_to_80': 8, '80_to_90': 9, '90_to_100': 10, 'NVG_surface': 11, 'buildings': 12,
            'low_vegetation': 13, 'playgrounds': 14, 'trees': 15, 'water': 16, 'no': 17, 'yes': 18, 'green_house': 19,
            'road': 20, 'bridge': 21, 'others': 22}

v2w_dict={0: '0', 1: '0_to_10', 2: '10_to_20', 3: '20_to_30', 4: '30_to_40', 5: '40_to_50', 6: '50_to_60',
            7: '60_to_70', 8: '70_to_80', 9: '80_to_90', 10: '90_to_100', 11: 'NVG_surface', 12: 'buildings',
            13: 'low_vegetation', 14: 'playgrounds', 15: 'trees', 16: 'water', 17: 'no', 18: 'yes', 19: 'green_house',
            20: 'road', 21: 'bridge', 22: 'others'}


def confuse_matrix(pre, label, n_class):
    pre = pre.cpu().numpy()
    label = label.cpu().numpy()
    cm = np.bincount(label * n_class + pre, minlength=n_class ** 2).reshape(n_class, n_class)
    return cm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=VisTA()
model = torch.nn.DataParallel(model).cuda()

model.module.load_state_dict(
    torch.load(
        './ckps/VisTA/VisTa.pth',
        map_location=device))

test_root = './test/'

val_dataset = DatasetNew(test_root+'json/Test.json',test_root+'image/',test_root+'mask/')
val_loader = DataLoader(val_dataset, batch_size=48, shuffle=False, num_workers=16, pin_memory=True)

model.eval()

cm_all = np.zeros((23, 23))
cm_nochange=np.zeros((2,2))

type_sum = {
    'change_ratio': [0, 0],  'change_or_not': [0, 0], 'change_to_what': [0, 0],
    'increase_or_not': [0, 0], 'decrease_or_not': [0, 0], 'smallest_change': [0, 0], 'largest_change': [0, 0],
    'change_from_what': [0, 0],
}

type_cm = {  # 0: TP, 1: FP, 2: FN
    'change_ratio': [0, 0, 0],  'change_or_not': [0, 0, 0], 'change_to_what': [0, 0, 0],
    'increase_or_not': [0, 0, 0], 'decrease_or_not': [0, 0, 0], 'smallest_change': [0, 0, 0],
    'largest_change': [0, 0, 0], 'change_from_what': [0, 0, 0],
}

ans_type = {
    'change_ratio': ['0', '0_to_10', '10_to_20', '20_to_30', '30_to_40', '40_to_50', '50_to_60', '60_to_70', '70_to_80',
                     '80_to_90', '90_to_100'],
    'change_or_not': ['yes', 'no'],
    'change_to_what': ['NVG_surface', 'buildings', 'low_vegetation', 'playgrounds', 'trees', 'water', 'green_house',
                       'road', 'bridge', 'others'],
    'increase_or_not': ['yes', 'no'],
    'decrease_or_not': ['yes', 'no'],
    'smallest_change': ['NVG_surface', 'buildings', 'low_vegetation', 'playgrounds', 'trees', 'water', 'green_house',
                        'road', 'bridge', 'others'],
    'largest_change': ['NVG_surface', 'buildings', 'low_vegetation', 'playgrounds', 'trees', 'water', 'green_house',
                       'road', 'bridge', 'others'],
    'change_from_what': ['NVG_surface', 'buildings', 'low_vegetation', 'playgrounds', 'trees', 'water', 'green_house',
                         'road', 'bridge', 'others'],
}

ans_type_sum = {
    'change_ratio': {'0': [0, 0], '0_to_10': [0, 0], '10_to_20': [0, 0], '20_to_30': [0, 0], '30_to_40': [0, 0],
                     '40_to_50': [0, 0], '50_to_60': [0, 0], '60_to_70': [0, 0], '70_to_80': [0, 0],
                     '80_to_90': [0, 0], '90_to_100': [0, 0]},
    'change_or_not': {'yes': [0, 0], 'no': [0, 0]},
    'change_to_what': {'NVG_surface': [0, 0], 'buildings': [0, 0], 'low_vegetation': [0, 0], 'playgrounds': [0, 0],
                       'trees': [0, 0], 'water': [0, 0], 'green_house': [0, 0],
                       'road': [0, 0], 'bridge': [0, 0], 'others': [0, 0]},
    'increase_or_not': {'yes': [0, 0], 'no': [0, 0]},
    'decrease_or_not': {'yes': [0, 0], 'no': [0, 0]},
    'smallest_change': {'NVG_surface': [0, 0], 'buildings': [0, 0], 'low_vegetation': [0, 0], 'playgrounds': [0, 0],
                        'trees': [0, 0], 'water': [0, 0], 'green_house': [0, 0],
                        'road': [0, 0], 'bridge': [0, 0], 'others': [0, 0]},
    'largest_change': {'NVG_surface': [0, 0], 'buildings': [0, 0], 'low_vegetation': [0, 0], 'playgrounds': [0, 0],
                       'trees': [0, 0], 'water': [0, 0], 'green_house': [0, 0],
                       'road': [0, 0], 'bridge': [0, 0], 'others': [0, 0]},
    'change_from_what': {'NVG_surface': [0, 0], 'buildings': [0, 0], 'low_vegetation': [0, 0], 'playgrounds': [0, 0],
                         'trees': [0, 0], 'water': [0, 0], 'green_house': [0, 0],
                         'road': [0, 0], 'bridge': [0, 0], 'others': [0, 0]},
}

tp = 0
fp = 0
fn = 0

for imgs1, imgs2, q_str, type_str, answer_vec, mask_img, a_id in tqdm(val_loader):
    imgs1 = imgs1.to(device)
    imgs2 = imgs2.to(device)
    answer_vec = answer_vec.to(device)
    mask_img = mask_img.to(device)

    with torch.no_grad():
        pred, ans = model(imgs1, imgs2, q_str, mask_img)

    pred[pred < 0.35] = 0
    pred[pred >= 0.35] = 1
    img_out = pred

    pred_img_max=np.max(img_out.cpu().numpy(), axis=(1, 2)).flatten()
    mask_img_max=np.max(mask_img.cpu().numpy(), axis=(2, 3)).flatten()
    cm_nochange[0][0]+=np.sum((pred_img_max==1) & (mask_img_max==1))
    cm_nochange[0][1]+=np.sum((pred_img_max==1) & (mask_img_max==0))
    cm_nochange[1][0]+=np.sum((pred_img_max==0) & (mask_img_max==1))
    cm_nochange[1][1]+=np.sum((pred_img_max==0) & (mask_img_max==0))

    pred = pred.flatten(1)
    target = mask_img.flatten(1)
    pred = pred.cpu().numpy()
    target = target.cpu().numpy()

    tp += np.sum((pred == 1) & (target == 1))
    fp += np.sum((pred == 1) & (target == 0))
    fn += np.sum((pred == 0) & (target == 1))

    ans = ans.argmax(dim=1)
    answer_vec = answer_vec.argmax(dim=1)
    cm_all += confuse_matrix(ans, answer_vec, 23)

    for i in range(len(type_str)):
        if type_str[i] == 'change_ratio_types':
            type_str[i] = 'change_ratio'
        type_cm[type_str[i]][0] += np.sum((pred[i] == 1) & (target[i] == 1))
        type_cm[type_str[i]][1] += np.sum((pred[i] == 1) & (target[i] == 0))
        type_cm[type_str[i]][2] += np.sum((pred[i] == 0) & (target[i] == 1))

    for i in range(len(type_str)):
        type_sum[type_str[i]][1] += 1
        ansvec=v2w_dict[int(answer_vec[i].cpu())]
        ans_type_sum[type_str[i]][ansvec][1] += 1
        if ans[i] == answer_vec[i]:
            type_sum[type_str[i]][0] += 1
            ans_type_sum[type_str[i]][ansvec][0] += 1

with open('cm.csv', 'w') as f:
    for i in range(23):
        for j in range(23):
            f.write(str(int(cm_all[i][j])) + ',')
        f.write('\n')

sum1 = 0
for key, value in type_sum.items():
    print('**** {} : {:.4f} {} {} ****'.format(key, value[0] / (value[1]+0.0001), value[0], value[1]))
    sum1 += value[0] / (value[1]+0.0001)
    for key1, value1 in ans_type_sum[key].items():
        print('{} : {:.4f} {} {}'.format(key1, value1[0] / (value1[1]+0.0001), value1[0], value1[1]))


OA = cm_all.diagonal().sum() / cm_all.sum()
print('OA:', OA)
print('AA:',sum1 / 8)

print('Precision:', tp / (tp + fp))

print('Recall:', tp / (tp + fn))
print('F1:', 2 * tp / (2 * tp + fp + fn))
print('IoU:', tp / (tp + fp + fn))
print('tp:', tp, 'fp:', fp, 'fn:', fn)

for key, value in type_cm.items():
    try:
        print('{} : Precision {:.4f} Recall {:.4f} F1 {:.4f} IoU={:.4f}'.format(key, value[0] / (value[0] + value[1]),
                                                                                value[0] / (value[0] + value[2]),
                                                                                2 * value[0] / (
                                                                                        2 * value[0] + value[1] + value[
                                                                                    2]),
                                                                                value[0] / (value[0] + value[1] + value[
                                                                                    2])))
    except:
        print(key, value)

p_0=cm_all.diagonal().sum() / cm_all.sum()
p_e=0
for i in range(23):
    p_e+=cm_all[i].sum()*cm_all[:,i].sum()
p_e/=cm_all.sum()*cm_all.sum()
k=(p_0-p_e)/(1-p_e)
print('Kappa:', k)

print('cm_nochange_TP:', cm_nochange[0][0])
print('cm_nochange_FP:', cm_nochange[0][1])
print('cm_nochange_FN:', cm_nochange[1][0])
print('cm_nochange_TN:', cm_nochange[1][1])