import pylibjpeg
import gc
import glob
import os
import re

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydicom as dicom
import torch
import torchvision as tv
from sklearn.model_selection import GroupKFold,StratifiedKFold,StratifiedGroupKFold
from torch.cuda.amp import GradScaler, autocast
from torchvision.models.feature_extraction import create_feature_extractor
from tqdm import tqdm

import wandb


def load_dicom(path):
    """
    This supports loading both regular and compressed JPEG images.
    See the first sell with `pip install` commands for the necessary dependencies
    """
    img = dicom.dcmread(path)
    img.PhotometricInterpretation = 'YBR_FULL'
    data = img.pixel_array
    data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return data, img


class EffnetDataSet(torch.utils.data.Dataset):
    def __init__(self, df, path, transforms=None):
        super().__init__()
        self.df = df
        self.path = path
        self.transforms = transforms
        self.n_25d_shift = 1
        self.n_25d_stride = 2


    def __getitem__(self, i):

        path = os.path.join(self.path, self.df.iloc[i].StudyInstanceUID, f'{self.df.iloc[i].Slice}.dcm')


        try:
            img = self.load_2_5d_slice(path)
            # Pytorch uses (batch, channel, height, width) order. Converting (height, width, channel) -> (channel, height, width)
            img = np.transpose(img, (2, 0, 1))
            if self.transforms is not None:
                img = self.transforms(torch.as_tensor(img))
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
        for i in range(-(self.n_25d_shift*self.n_25d_stride), self.n_25d_shift*self.n_25d_stride +1,self.n_25d_stride):  # eg: i = {-2, -1, 0, 1, 2}
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

class EffnetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        effnet = tv.models.resnext50_32x4d(weights=WEIGHTS)
        self.model = create_feature_extractor(effnet, ['flatten'])
        self.nn_fracture = torch.nn.Sequential(
            torch.nn.Linear(2048, 7),
        )
        self.nn_vertebrae = torch.nn.Sequential(
            torch.nn.Linear(2048, 7),
        )

    def forward(self, x):
        # returns logits
        x = self.model(x)['flatten']
        return self.nn_fracture(x), self.nn_vertebrae(x)

    def predict(self, x):
        frac, vert = self.forward(x)
        return torch.sigmoid(frac), torch.sigmoid(vert)

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
def filter_nones(b):
    return torch.utils.data.default_collate([v for v in b if v is not None])
def save_model(name, model):
    torch.save(model.state_dict(), f'{name}.tph')
def load_model(model, name, path='.'):
    data = torch.load(os.path.join(path, f'{name}.tph'), map_location=DEVICE)
    model.load_state_dict(data)
    return model


def evaluate_effnet(model: EffnetModel, ds, max_batches=100000, shuffle=False):
    torch.manual_seed(42)
    model = model.to(DEVICE)
    dl_test = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=4,
                                          collate_fn=filter_nones)
    pred_frac = []
    pred_vert = []
    with torch.no_grad():
        model.eval()
        frac_losses = []
        vert_losses = []
        with tqdm(dl_test, desc='Eval', miniters=10) as progress:
            for i, (X, y_frac, y_vert) in enumerate(progress):
                with autocast():
                    y_frac_pred, y_vert_pred = model.forward(X.to(DEVICE))
                    frac_loss = weighted_loss(y_frac_pred, y_frac.to(DEVICE)).item()
                    vert_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_vert_pred, y_vert.to(DEVICE)).item()
                    pred_frac.append(torch.sigmoid(y_frac_pred))
                    pred_vert.append(torch.sigmoid(y_vert_pred))
                    frac_losses.append(frac_loss)
                    vert_losses.append(vert_loss)

                if i >= max_batches:
                    break
        return np.mean(frac_losses), np.mean(vert_losses), torch.concat(pred_frac).cpu().numpy(), torch.concat(pred_vert).cpu().numpy()

def gc_collect():
    gc.collect()
    torch.cuda.empty_cache()

def train_effnet(ds_train, ds_eval, logger, name, loss_weight):
    torch.manual_seed(42)
    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,
                                           collate_fn=filter_nones)

    model = EffnetModel().to(DEVICE)
    optim = torch.optim.Adam(model.parameters())
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=ONE_CYCLE_MAX_LR, epochs=1,
                                                    steps_per_epoch=min(EFFNET_MAX_TRAIN_BATCHES, len(dl_train)),
                                                    pct_start=ONE_CYCLE_PCT_START)

    # scheduler = torch.optim.lr_scheduler.CyclicLR(optim,base_lr=1e-6,max_lr=8e-5,step_size_up=2000,step_size_down=2000,
    #                                               mode='triangular2',cycle_momentum=False)

    model.train()
    scaler = GradScaler()
    with tqdm(dl_train, desc='Train', miniters=10) as progress:
        for batch_idx, (X, y_frac, y_vert) in enumerate(progress):

            if ds_eval is not None and batch_idx % SAVE_CHECKPOINT_EVERY_STEP == 0 and EFFNET_MAX_EVAL_BATCHES > 0:
                frac_loss, vert_loss = evaluate_effnet(
                    model, ds_eval, max_batches=EFFNET_MAX_EVAL_BATCHES, shuffle=True)[:2]
                model.train()
                logger.log(
                    {'eval_frac_loss': frac_loss, 'eval_vert_loss': vert_loss, 'eval_loss': frac_loss + vert_loss})
                if batch_idx > 0:  # don't save untrained model
                    save_model(name, model)

            if batch_idx >= EFFNET_MAX_TRAIN_BATCHES:
                break

            # if batch_idx % 4000 ==0 and batch_idx >0:
            #     loss_weight *= 2

            optim.zero_grad()
            # Using mixed precision training
            with autocast():
                y_frac_pred, y_vert_pred = model.forward(X.to(DEVICE))
                frac_loss = weighted_loss(y_frac_pred, y_frac.to(DEVICE))
                vert_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_vert_pred, y_vert.to(DEVICE))
                loss = (loss_weight * frac_loss + vert_loss)

                if np.isinf(loss.item()) or np.isnan(loss.item()):
                    print(f'Bad loss, skipping the batch {batch_idx}')
                    del loss, frac_loss, vert_loss, y_frac_pred, y_vert_pred
                    gc_collect()
                    continue

            # scaler is needed to prevent "gradient underflow"
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            scheduler.step()

            progress.set_description(f'Train loss: {loss.item() :.02f}')
            logger.log({'loss': (loss.item()), 'frac_loss': frac_loss.item(), 'vert_loss': vert_loss.item(),
                        'lr': scheduler.get_last_lr()[0]})

    save_model(name, model)
    return model

if __name__ == '__main__':


    plt.rcParams['figure.figsize'] = (20, 5)
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 1000)

    # Effnet
    WEIGHTS = tv.models.ResNeXt50_32X4D_Weights.DEFAULT
    RSNA_2022_PATH = '../input/rsna-2022-cervical-spine-fracture-detection'
    TRAIN_IMAGES_PATH = f'{RSNA_2022_PATH}/train_images'
    TEST_IMAGES_PATH = f'{RSNA_2022_PATH}/test_images'
    EFFNET_MAX_TRAIN_BATCHES = 8000
    EFFNET_MAX_EVAL_BATCHES = 200
    ONE_CYCLE_MAX_LR = 0.00008
    ONE_CYCLE_PCT_START = 0.3
    SAVE_CHECKPOINT_EVERY_STEP = 2000
    EFFNET_CHECKPOINTS_PATH = '../output/ckpt-v2'
    FRAC_LOSS_WEIGHT = 0.2
    N_FOLDS = 4
    METADATA_PATH = '../input/vertebrae-detection-checkpoints'

    PREDICT_MAX_BATCHES = 1e9

    # Common

    IS_KAGGLE = False

    os.environ["WANDB_MODE"] = "online"



    print('Running locally')
    RSNA_2022_PATH = '../input/rsna-2022-cervical-spine-fracture-detection'
    TRAIN_IMAGES_PATH = '../input/rsna-2022-cervical-spine-fracture-detection/train_images'
    TEST_IMAGES_PATH = '../input/rsna-2022-cervical-spine-fracture-detection/test_images'
    METADATA_PATH = r'W:\PycharmProjects\kaggle-RSNA\input\rsna-2022-spine-fracture-detection-metadata'
    EFFNET_CHECKPOINTS_PATH = 'frac_checkpoints'
    os.environ['WANDB_API_KEY'] = '38c8eb6b3d0a6b9221c841d0d67ca44a78c4190e'

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    if DEVICE == 'cuda':
        BATCH_SIZE = 64
    else:
        BATCH_SIZE = 2
    df_train = pd.read_csv(f'{RSNA_2022_PATH}/train.csv')
    # rsna-2022-spine-fracture-detection-metadata contains inference of C1-C7 vertebrae for all training sample (95% accuracy)
    df_train_slices = pd.read_csv(f'{METADATA_PATH}/train_segmented.csv')
    c1c7 = [f'C{i}' for i in range(1, 8)]
    df_train_slices[c1c7] = (df_train_slices[c1c7] > 0.5).astype(int)
    df_train = df_train_slices.set_index('StudyInstanceUID').join(df_train.set_index('StudyInstanceUID'),
                                                                  rsuffix='_fracture').reset_index().copy()
    df_train = df_train.query('StudyInstanceUID != "1.2.826.0.1.3680043.20574"').reset_index(drop=True)
    split = StratifiedGroupKFold(N_FOLDS)
    for k, (_, test_idx) in enumerate(split.split(df_train, groups=df_train.StudyInstanceUID,y=df_train.patient_overall)):
        df_train.loc[test_idx, 'split'] = k

    df_test = pd.read_csv(f'{RSNA_2022_PATH}/test.csv')

    if df_test.iloc[0].row_id == '1.2.826.0.1.3680043.10197_C1':
        # test_images and test.csv are inconsistent in the dev dataset, fixing labels for the dev run.
        df_test = pd.DataFrame({
            "row_id": ['1.2.826.0.1.3680043.22327_C1', '1.2.826.0.1.3680043.25399_C1', '1.2.826.0.1.3680043.5876_C1'],
            "StudyInstanceUID": ['1.2.826.0.1.3680043.22327', '1.2.826.0.1.3680043.25399', '1.2.826.0.1.3680043.5876'],
            "prediction_type": ["C1", "C1", "patient_overall"]}
        )



    # N-fold models. Can be used to estimate accurate CV score and in ensembled submissions.
    effnet_models = []
    for fold in range(N_FOLDS):
        if os.path.exists(os.path.join(EFFNET_CHECKPOINTS_PATH, f'effnetv2-f{fold}.tph')):
            print(f'Found cached version of effnetv2-f{fold}')
            effnet_models.append(load_model(EffnetModel(), f'effnetv2-f{fold}', EFFNET_CHECKPOINTS_PATH))
        else:
            with wandb.init(project='RSNA-2022', name=f'EffNet-v2-fold{fold}') as run:
                gc_collect()
                ds_train = EffnetDataSet(df_train.query('split != @fold'), TRAIN_IMAGES_PATH,tv.transforms.Resize((384,384)))
                ds_eval = EffnetDataSet(df_train.query('split == @fold'), TRAIN_IMAGES_PATH,tv.transforms.Resize((384,384)))
                effnet_models.append(train_effnet(ds_train, ds_eval, run, f'effnetv2-f{fold}',FRAC_LOSS_WEIGHT))

    # "Main" model that uses all folds data. Can be used in single-model submissions.
    if os.path.exists(os.path.join(EFFNET_CHECKPOINTS_PATH, f'effnetv2.tph')):
        print(f'Found cached version of effnetv2')
        effnet_models.append(load_model(EffnetModel(), f'effnetv2', EFFNET_CHECKPOINTS_PATH))
    else:
        with wandb.init(project='RSNA-2022', name=f'EffNet-v2') as run:
            gc_collect()
            ds_train = EffnetDataSet(df_train, TRAIN_IMAGES_PATH, WEIGHTS.transforms())
            train_effnet(ds_train, None, run, f'effnetv2')
