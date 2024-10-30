import os
import cv2
import torch
import numpy as np
import datetime
import scipy.io as sio
from utils.train_transunet import TransUNetSeg
from utils.utils import thresh_func
from utils.config import cfg


class Inference:
    def __init__(self, model_path, device):
        self.device = device
        self.transunet = TransUNetSeg(device)
        self.transunet.load_model(model_path)

        if not os.path.exists('./results'):
            os.mkdir('./results')

    def read_and_preprocess(self, p):
        img=sio.loadmat(p)
        img=img['img']
        img_torch=(img-np.min(np.abs(img)))/(np.max(np.abs(img))-np.min(np.abs(img)))
        img_real = np.real(img_torch)  # 实部
        img_imag = np.imag(img_torch)  # 虚部
        img_torch = np.stack((img_real, img_imag), axis=-1)  # 将实部和虚部沿最后一个维度（通道维度）叠加，形成二通道数据
        # 调整图像和掩码大小
        img_torch = cv2.resize(img_torch, (cfg.transunet.img_dim, cfg.transunet.img_dim))
        img_torch = img_torch.transpose((2, 0, 1))
        img_torch = np.expand_dims(img_torch, axis=0)
        img_torch = torch.from_numpy(img_torch.astype('float32')).to(self.device)
        return img, img_torch

    def save_preds(self, preds):
        folder_path = './results/' + str(datetime.datetime.utcnow()).replace(' ', '_').replace(':', '-')
        os.mkdir(folder_path)
        print(preds)
        for name, pred_mask in preds.items():
            cv2.imwrite(f'{folder_path}/{name}', pred_mask)

    def infer(self, path, merged=False, save=True):
        path = [path] if isinstance(path, str) else path

        preds = {}
        for p in path:
            file_name = p.split('/')[-1]
            img, img_torch = self.read_and_preprocess(p)
            with torch.no_grad():
                pred_mask = self.transunet.model(img_torch)
                pred_mask = torch.sigmoid(pred_mask)
                pred_mask = pred_mask.detach().cpu().numpy().transpose((0, 2, 3, 1))

            orig_h, orig_w = img.shape[:2]
            pred_mask = cv2.resize(pred_mask[0, ...], (orig_w, orig_h))
            pred_mask = thresh_func(pred_mask, thresh=cfg.inference_threshold)
            pred_mask *= 255

            if merged:
                pred_mask = cv2.bitwise_and(img, img, mask=pred_mask.astype('uint8'))

            preds[file_name] = pred_mask

        if save:
            self.save_preds(preds)

        return preds
