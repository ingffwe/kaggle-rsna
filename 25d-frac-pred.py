import os
import re
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom as dicom
from sklearn.model_selection import train_test_split
import torch
import torchvision as tv
from sklearn.model_selection import GroupKFold
from torch.cuda.amp import GradScaler, autocast
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm
import timm

def load_dicom(path):
    """
    This supports loading both regular and compressed JPEG images.
    See the first sell with `pip install` commands for the necessary dependencies
    """
    img=dicom.dcmread(path)
    img.PhotometricInterpretation = 'YBR_FULL'
    data = img.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data=(data * 255).astype(np.uint8)
    return data, img


RSNA_2022_PATH = '../input/rsna-2022-cervical-spine-fracture-detection'
TRAIN_IMAGES_PATH = f'{RSNA_2022_PATH}/train_images'
TEST_IMAGES_PATH = f'{RSNA_2022_PATH}/test_images'
EFFNET_MAX_TRAIN_BATCHES = 40000
EFFNET_MAX_EVAL_BATCHES = 200
ONE_CYCLE_MAX_LR = 0.0001
ONE_CYCLE_PCT_START = 0.3
SAVE_CHECKPOINT_EVERY_STEP = 1000
EFFNET_CHECKPOINTS_PATH = '../input/rsna-2022-base-effnetv2'
FRAC_LOSS_WEIGHT = 2.
N_FOLDS = 5
METADATA_PATH = '../input/vertebrae-detection-checkpoints'

PREDICT_MAX_BATCHES = 1e7

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    BATCH_SIZE = 32
else:
    BATCH_SIZE = 2

DEVICE


class EffnetDataSet(torch.utils.data.Dataset):
    def __init__(self, df, path, transforms=None):
        super().__init__()
        self.df = df
        self.path = path
        self.transforms = transforms
        self.n_25d_shift = 1


    def __getitem__(self, i):

        path = os.path.join(self.path, self.df.iloc[i].StudyInstanceUID, f'{self.df.iloc[i].Slice}.dcm')


        try:
            img = self.load_2_5d_slice(path)
            # Pytorch uses (batch, channel, height, width) order. Converting (height, width, channel) -> (channel, height, width)
            # img = np.transpose(img, (2, 0, 1))
            if self.transforms is not None:
                img = self.transforms(image=img)['image']
            img = img.to(dtype=torch.float16)
        except Exception as ex:
            print(ex)
            return None

        if 'C1_fracture' in self.df:
            frac_targets = torch.as_tensor(self.df.iloc[i][['C1_fracture', 'C2_fracture', 'C3_fracture', 'C4_fracture',
                                                            'C5_fracture', 'C6_fracture', 'C7_fracture']].astype(
                'float32').values)
            vert_targets = torch.as_tensor(
                self.df.iloc[i][['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']].astype('float32').values)
            frac_targets = frac_targets * vert_targets  # we only enable targets that are visible on the current slice
            return img, frac_targets, vert_targets
        return img

    def __len__(self):
        return len(self.df)

    def load_2_5d_slice(self, middle_img_path):
        #### 步骤1: 获取中间图片的基本信息
        #### eg: middle_img_path: '5.dcm'
        middle_slice_num = os.path.basename(middle_img_path).split('.')[0]  # eg: 5
        middle_str = str(middle_slice_num)
        # img = load_dicom(middle_img_path)[0]

        new_25d_imgs = []

        ##### 步骤2：按照左右n_25d_shift数量进行填充，如果没有相应图片填充为Nan.
        ##### 注：经过EDA发现同一天的所有患者图片的shape是一致的
        for i in range(-self.n_25d_shift, self.n_25d_shift + 1):  # eg: i = {-2, -1, 0, 1, 2}
            shift_slice_num = int(middle_slice_num) + i
            shift_str = str(shift_slice_num)
            shift_img_path = middle_img_path.replace(middle_str, shift_str)

            if os.path.exists(shift_img_path):
                shift_img = load_dicom(shift_img_path)[0]
                new_25d_imgs.append(shift_img)
            else:
                new_25d_imgs.append(None)

        ##### 步骤3：从中心开始往外循环，依次填补None的值
        ##### eg: n_25d_shift = 2, 那么形成5个channel, idx为[0, 1, 2, 3, 4], 所以依次处理的idx为[1, 3, 0, 4]
        shift_left_idxs = []
        shift_right_idxs = []
        for related_idx in range(self.n_25d_shift):
            shift_left_idxs.append(self.n_25d_shift - related_idx - 1)
            shift_right_idxs.append(self.n_25d_shift + related_idx + 1)

        for left_idx, right_idx in zip(shift_left_idxs, shift_right_idxs):
            if new_25d_imgs[left_idx] is None:
                new_25d_imgs[left_idx] = new_25d_imgs[left_idx + 1]
            if new_25d_imgs[right_idx] is None:
                new_25d_imgs[right_idx] = new_25d_imgs[right_idx - 1]
        try:
            new_25d_imgs = np.stack(new_25d_imgs, axis=2).astype('float32')  # [w, h, c]
            mx_pixel = new_25d_imgs.max()
            if mx_pixel != 0:
                new_25d_imgs /= mx_pixel
        except:
            return np.zeros((512, 512, 3))
        return new_25d_imgs


df_train = pd.read_csv(f'{RSNA_2022_PATH}/train.csv')
df_train_slices = pd.read_csv(r'W:\PycharmProjects\kaggle-RSNA\input\rsna-2022-spine-fracture-detection-metadata\train_segmented.csv')
c1c7 = [f'C{i}' for i in range(1, 8)]
df_train_slices[c1c7] = (df_train_slices[c1c7] > 0.5).astype(int)
df_train = df_train_slices.set_index('StudyInstanceUID').join(df_train.set_index('StudyInstanceUID'),
                                                              rsuffix='_fracture').reset_index().copy()
df_train = df_train.query('StudyInstanceUID != "1.2.826.0.1.3680043.20574"').reset_index(drop=True)

split = GroupKFold(N_FOLDS)
for k, (_, test_idx) in enumerate(split.split(df_train, groups=df_train.StudyInstanceUID)):
    df_train.loc[test_idx, 'split'] = k



def weighted_loss(y_pred_logit, y, reduction='mean', verbose=False):
    """
    Weighted loss
    We reuse torch.nn.functional.binary_cross_entropy_with_logits here. pos_weight and weights combined give us necessary coefficients described in https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/340392

    See also this explanation: https://www.kaggle.com/code/samuelcortinhas/rsna-fracture-detection-in-depth-eda/notebook
    """

    neg_weights = (torch.tensor([7., 1, 1, 1, 1, 1, 1, 1]) if y_pred_logit.shape[-1] == 8 else torch.ones(y_pred_logit.shape[-1])).to(DEVICE)
    pos_weights = (torch.tensor([14., 2, 2, 2, 2, 2, 2, 2]) if y_pred_logit.shape[-1] == 8 else torch.ones(y_pred_logit.shape[-1]) * 2.).to(DEVICE)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        y_pred_logit,
        y,
        reduction='none',
    )

    if verbose:
        print('loss', loss)

    pos_weights = y * pos_weights.unsqueeze(0)
    neg_weights = (1 - y) * neg_weights.unsqueeze(0)
    all_weights = pos_weights + neg_weights

    if verbose:
        print('all weights', all_weights)

    loss *= all_weights
    if verbose:
        print('weighted loss', loss)

    norm = torch.sum(all_weights, dim=1).unsqueeze(1)
    if verbose:
        print('normalization factors', norm)

    loss /= norm
    if verbose:
        print('normalized loss', loss)

    loss = torch.sum(loss, dim=1)
    if verbose:
        print('summed up over patient_overall-C1-C7 loss', loss)

    if reduction == 'mean':
        return torch.mean(loss)
    return loss


class EffnetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        effnet = tv.models.efficientnet_v2_s()
        self.model = create_feature_extractor(effnet, ['flatten'])
        self.nn_fracture = torch.nn.Sequential(
            torch.nn.Linear(1280, 7),
        )
        self.nn_vertebrae = torch.nn.Sequential(
            torch.nn.Linear(1280, 7),
        )

    def forward(self, x):
        # returns logits
        x = self.model(x)['flatten']
        return self.nn_fracture(x), self.nn_vertebrae(x)

    def predict(self, x):
        frac, vert = self.forward(x)
        return torch.sigmoid(frac), torch.sigmoid(vert)


def filter_nones(b):
    return torch.utils.data.default_collate([v for v in b if v is not None])

def evaluate_effnet(model: EffnetModel, dl_test, max_batches=PREDICT_MAX_BATCHES, shuffle=False):
    torch.manual_seed(42)
    model = model.to(DEVICE)
    # dl_test = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=os.cpu_count(),
                                          # collate_fn=filter_nones)
    pred_frac = []
    pred_vert = []
    with torch.no_grad():
        model.eval()
        frac_losses = []

        with tqdm(dl_test, desc='Eval') as progress:
            for i, (X, y_frac, y_vert) in enumerate(progress):
                with autocast():
                    y_frac_pred = model(X.to(DEVICE))
                    frac_loss = weighted_loss(y_frac_pred, y_frac.to(DEVICE)).item()
                    pred_frac.append(torch.sigmoid(y_frac_pred))
                    pred_vert.append(y_vert)
                    frac_losses.append(frac_loss)

                if i >= max_batches:
                    break
        return np.mean(frac_losses), torch.concat(pred_frac).cpu().numpy(),torch.concat(pred_vert).cpu().numpy()

def patient_prediction(df):
    c1c7 = np.average(df[frac_cols].values, axis=0, weights=df[vert_cols].values)
    pred_patient_overall = 1 - np.prod(1 - c1c7)
    return np.concatenate([[pred_patient_overall], c1c7])


if __name__ == '__main__':
    fold = 0
    ds_train = EffnetDataSet(df_train.query('split != @fold'), TRAIN_IMAGES_PATH, A.Compose([
        A.Resize(512, 512),
        A.HorizontalFlip(p=0.5),
        # # A.RandomContrast(p=0.5),
        # # A.RandomBrightness(p=0.5),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        # # A.RandomBrightness(limit=2, p=0.5),
        # A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.05, rotate_limit=10, p=0.2),
        #
        # A.OneOf([
        #     A.MotionBlur(p=0.2),
        #     A.MedianBlur(blur_limit=3, p=0.1),
        #     A.Blur(blur_limit=3, p=0.1),
        # ], p=0.5),
        ToTensorV2(),

    ])
                             )

    ds_val = EffnetDataSet(df_train.query('split == @fold'), TRAIN_IMAGES_PATH, A.Compose([
        A.Resize(384, 384),
        # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
                           )

    # train_loader = torch.utils.data.DataLoader(ds_train, batch_size=16, shuffle=True, num_workers=10)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=64, shuffle=False, num_workers=8)

    model = torch.jit.load('../output/ckpt-25d-frac/ENV2_384_fold0_loss0.0404.pth')
    model = model.to('cuda')

    frac_loss, effnet_pred_frac, effnet_pred_vert = evaluate_effnet(model, val_loader, PREDICT_MAX_BATCHES)

    df_pred = []
    fold = 0
    effnet_pred_vert = effnet_pred_frac
    df_effnet_pred = pd.DataFrame(data=np.concatenate([effnet_pred_frac, effnet_pred_vert], axis=1),
                                  columns=[f'C{i}_effnet_frac' for i in range(1, 8)] +
                                          [f'C{i}_effnet_vert' for i in range(1, 8)])
    df = pd.concat([df_train.query('split == @fold').head(len(df_effnet_pred)).reset_index(drop=True), df_effnet_pred],
                   axis=1).sort_values(['StudyInstanceUID', 'Slice'])
    df_pred.append(df)
    df_pred = pd.concat(df_pred)

    target_cols = ['patient_overall'] + [f'C{i}_fracture' for i in range(1, 8)]
    frac_cols = [f'C{i}_effnet_frac' for i in range(1, 8)]
    vert_cols = [f'C{i}_effnet_vert' for i in range(1, 8)]

    df_patient_pred = df_pred.groupby('StudyInstanceUID').apply(lambda df: patient_prediction(df)).to_frame(
        'pred').join(df_pred.groupby('StudyInstanceUID')[target_cols].mean())

    predictions = np.stack(df_patient_pred.pred.values.tolist())
    targets = df_patient_pred[target_cols].values
    print('CV score:', weighted_loss(torch.logit(torch.as_tensor(predictions)).to(DEVICE), torch.as_tensor(targets).to(DEVICE)))
