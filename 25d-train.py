import gc
import glob
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
    # return cv2.cvtColor(data, cv2.COLOR_GRAY2RGB), img
    return data, img




# Effnet
# WEIGHTS = tv.models.efficientnet.EfficientNet_V2_S_Weights.DEFAULT
RSNA_2022_PATH = '../input/rsna-2022-cervical-spine-fracture-detection'
TRAIN_IMAGES_PATH = f'{RSNA_2022_PATH}/train_images'
TEST_IMAGES_PATH = f'{RSNA_2022_PATH}/test_images'
EFFNET_MAX_TRAIN_BATCHES = 40000
EFFNET_MAX_EVAL_BATCHES = 200
ONE_CYCLE_MAX_LR = 0.0003
ONE_CYCLE_PCT_START = 0.3
SAVE_CHECKPOINT_EVERY_STEP = 1000
EFFNET_CHECKPOINTS_PATH = '../input/rsna-2022-base-effnetv2'
FRAC_LOSS_WEIGHT = 2.
N_FOLDS = 5
METADATA_PATH = '../input/vertebrae-detection-checkpoints'

PREDICT_MAX_BATCHES = 1e9

DEVICE='cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    BATCH_SIZE = 32
else:
    BATCH_SIZE = 2


##############################################################
#################### df
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

# df_test = pd.read_csv(f'{RSNA_2022_PATH}/test.csv')
#
# if df_test.iloc[0].row_id == '1.2.826.0.1.3680043.10197_C1':
#     # test_images and test.csv are inconsistent in the dev dataset, fixing labels for the dev run.
#     df_test = pd.DataFrame({
#         "row_id": ['1.2.826.0.1.3680043.22327_C1', '1.2.826.0.1.3680043.25399_C1', '1.2.826.0.1.3680043.5876_C1'],
#         "StudyInstanceUID": ['1.2.826.0.1.3680043.22327', '1.2.826.0.1.3680043.25399', '1.2.826.0.1.3680043.5876'],
#         "prediction_type": ["C1", "C1", "patient_overall"]}
#     )
#
# test_slices = glob.glob(f'{TEST_IMAGES_PATH}/*/*')
# test_slices = [re.findall(f'{TEST_IMAGES_PATH}/(.*)/(.*).dcm', s)[0] for s in test_slices]
# df_test_slices = pd.DataFrame(data=test_slices, columns=['StudyInstanceUID', 'Slice'])
# df_test = df_test.set_index('StudyInstanceUID').join(df_test_slices.set_index('StudyInstanceUID')).reset_index()


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


class EffnetDataSet(torch.utils.data.Dataset):
    def __init__(self, df, path, transforms=None):
        super().__init__()
        self.df = df
        self.path = path
        self.transforms = transforms
        self.n_25d_shift = 1
        self.n_25d_stride = 1



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


def train(train_loader, model, optimizer):

    model.train()
    scaler = GradScaler()
    train_loss = 0.0
    pbar = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train ')
    best_loss = 100.0
    for batch_idx, (X, y_frac,y_vert) in pbar:


        optimizer.zero_grad()
        with autocast():
            y_frac_pred, y_vert_pred = model.forward(X.to(DEVICE))
            frac_loss = weighted_loss(y_frac_pred, y_frac.to(DEVICE))
            vert_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_vert_pred, y_vert.to(DEVICE))
            loss = FRAC_LOSS_WEIGHT * frac_loss + vert_loss

            if np.isinf(loss.item()) or np.isnan(loss.item()):
                print(f'Bad loss, skipping the batch {batch_idx}')
                del loss, frac_loss, vert_loss, y_frac_pred, y_vert_pred
                continue

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        train_loss += loss

        if batch_idx % 1000==0 and batch_idx!=0:
            new_loss = train_loss/1000
            train_loss = 0
            print(new_loss)
            if new_loss < best_loss:
                torch.save(model.state_dict(), f'../output/ckpt-25d/ENV2_1013_512_loss{new_loss:.4f}.tph')
                best_loss = new_loss
            print({'loss': (loss.item()), 'frac_loss': frac_loss.item(), 'vert_loss': vert_loss.item(),
                    'lr': scheduler.get_last_lr()[0]})

    print("Train loss :"+str(train_loss))
    return train_loss/len(train_loader)




def validate(val_loader, model):

    pred_frac = []
    pred_vert = []
    with torch.no_grad():
        frac_losses = []
        vert_losses = []
        model.eval()
        pbar = tqdm(enumerate(val_loader), total=len(val_loader), desc='Val ')

        for batch_idx, (X, y_frac, y_vert) in pbar:
            with autocast():
                y_frac_pred, y_vert_pred = model.forward(X.to(DEVICE))
                frac_loss = weighted_loss(y_frac_pred, y_frac.to(DEVICE)).item()
                vert_loss = torch.nn.functional.binary_cross_entropy_with_logits(y_vert_pred, y_vert.to(DEVICE)).item()
                pred_frac.append(torch.sigmoid(y_frac_pred))
                pred_vert.append(torch.sigmoid(y_vert_pred))
                frac_losses.append(frac_loss)
                vert_losses.append(vert_loss)

        frac_loss,vert_loss = np.mean(frac_losses), np.mean(vert_losses)
    print("Val loss :"+ str(vert_loss+frac_loss))
    return vert_loss+frac_loss


def predict(test_loader, model, criterion):
    model.eval()
    val_acc = 0.0

    test_pred = []
    with torch.no_grad():
        # end = time.time()
        for i, (input, target) in enumerate(test_loader):
            input = input.cuda()
            target = target.cuda()

            # compute output
            output = model(input)
            test_pred.append(output.data.cpu().numpy())

    return np.vstack(test_pred)


if __name__ == '__main__':

    # train_df, val_df = train_test_split(df_train, test_size=0.2, random_state=42, stratify = df_train.patient_overall)
    fold = 0
    ds_train = EffnetDataSet(df_train.query('split != @fold'), TRAIN_IMAGES_PATH, A.Compose([
                          A.Resize(384, 384),
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

    ds_val = EffnetDataSet(df_train.query('split == @fold'), TRAIN_IMAGES_PATH,A.Compose([
                          A.Resize(384, 384),
                          # A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                          ToTensorV2(),
                          ])
                      )

    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=16, shuffle=True, num_workers=10)
    val_loader = torch.utils.data.DataLoader(ds_val, batch_size=16, shuffle=True, num_workers=8)

    # model = timm.create_model("tf_efficientnetv2_l_in21ft1k",
    #                           num_classes=7,
    #                           in_chans=3,
    #                           pretrained=True)

    model = EffnetModel()

    model.load_state_dict(torch.load('../output/ckpt-25d/ENV2_1013_512_loss0.2155.tph'))
    model = model.to('cuda')

    # optimizer = torch.optim.SGD(model.parameters(), 0.005)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=5, mode="triangular2")

    optimizer = torch.optim.Adam(model.parameters())
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.001, max_lr=0.01, step_size_up=5, mode="triangular2")
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 8)
    # best_acc = validate(val_loader, model, criterion)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=ONE_CYCLE_MAX_LR, epochs=3,
                                                    steps_per_epoch=min(EFFNET_MAX_TRAIN_BATCHES, len(train_loader)),
                                                    pct_start=ONE_CYCLE_PCT_START)
    best_acc = 0
    best_loss = 100

    for _ in range(5):
        print('epoch:' + str(_ + 1))

        train_loss = train(train_loader, model, optimizer)
        # val_loss = validate(val_loader, model)


        #
        # if val_loss < best_loss:
        #     torch.save(model.state_dict(), '../output/ckpt/ENV2_test.pth')
        #     best_loss = val_loss

        # scheduler.step()
        # print(train_loss,val_loss)



