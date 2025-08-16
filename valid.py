# valid.py

import os, time, yaml, shutil, logging, argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
from sklearn.metrics import confusion_matrix

import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from medpy.metric.binary import hd95 as medpy_hd95

from utils.image_dataset              import Image_Dataset
from utils.tools                      import seed_reproducer, load_checkpoint
from utils.get_logger                 import open_log
from networks.embedding_translation   import MainPipeline
from networks.vae_setup.autoencoder   import AutoencoderKL
from networks.vae_setup.distributions import DiagonalGaussianDistribution
from utils.vis_utils                  import save_imgs, save_predicted_mask, save_all_individually

def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    pred = pred.astype(bool)
    gt   = gt.astype(bool)
    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if (pred.sum() == 0) ^ (gt.sum() == 0):
        h, w = gt.shape
        return float(np.hypot(h, w))
    return medpy_hd95(pred, gt)

def get_mu_sigma(post, scale):
    mu, logvar = post.mu_and_sigma()
    return scale * mu, logvar

def vae_decode(vae, mu_pred, scale):
    z   = (mu_pred / scale).clamp(-10, 10)
    seg = vae.decode(z).mean(1, True).clamp(-1, 1)
    seg = torch.nan_to_num(seg, nan=0.0, posinf=1.0, neginf=-1.0)
    return ((seg + 1) * 0.5).clamp(0, 1)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='./configs/valid.yaml')
    p.add_argument('--vis', choices=['composite', 'separate', 'pred'],
                   default=None,
                   help="If set, will save visual outputs (composite / separate / prediction mask)")
    return p.parse_args()

def main():
    args = parse_args()
    cfg  = yaml.safe_load(open(args.config))

    pred_dir = cfg['save_seg_img_path']
    if os.path.exists(pred_dir):
        shutil.rmtree(pred_dir)
    os.makedirs(pred_dir, exist_ok=True)

    Path(cfg['snapshot_path']).mkdir(parents=True, exist_ok=True)
    cfg['log_path'] = os.path.join(cfg['snapshot_path'], 'logs')
    Path(cfg['log_path']).mkdir(parents=True, exist_ok=True)

    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, cfg['GPUs']))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_reproducer(cfg['seed'])
    open_log(args, cfg)

    # data loader
    valid_loader = DataLoader(
        Image_Dataset(cfg['pickle_file_path'], 'test'),
        batch_size=1, shuffle=False, drop_last=False, pin_memory=True
    )

    # main pipeline
    mapper = MainPipeline(
        in_channel   = cfg['in_channel'],
        out_channels = cfg['out_channels'],
        ch           = cfg['ch'],
        ch_mult      = tuple(cfg['ch_mult'])
    )
    mapper = load_checkpoint(mapper, cfg['model_weight'])
    mapper.to(device).eval()

    # VAE
    vae_cfg = OmegaConf.load(cfg['vae_config_path'])
    vae = AutoencoderKL(**vae_cfg.first_stage_config.params)
    vae.load_state_dict(
        torch.load(cfg['vae_ckpt'], map_location='cpu')['state_dict'],
        strict=True
    )
    vae.eval().requires_grad_(False).to(device)
    scale = vae_cfg.first_stage_config.scale_factor

    names, dscs, ious, hd95s = [], [], [], []
    ds_name = cfg.get('dataset', 'generic')

    t0 = time.time()
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(valid_loader, desc='Validate')):
            original_name = batch['name'][0].replace('_segmentation', '')
            names.append(original_name)

            img      = (2*batch['img'] - 1).to(device)
            seg_raw  = (batch['seg'].permute(0,3,1,2) / 255.).to(device)
            seg_bin  = seg_raw.mean(1, True)

            mu,_     = get_mu_sigma(vae.encode(img), scale)
            prob     = vae_decode(vae, mapper(mu)['out'], scale)

            pred = (prob >= 0.5).cpu().numpy().astype(np.uint8).squeeze()
            gt   = (seg_bin >= 0.5).cpu().numpy().astype(np.uint8).squeeze()

            # confusion terms for DSC/IoU
            tn, fp, fn, tp = confusion_matrix(
                gt.ravel(), pred.ravel(), labels=[0,1]
            ).ravel()

            # metrics
            denom_dsc = (2*tp + fp + fn)
            denom_iou = (tp + fp + fn)
            dsc = 2*tp/denom_dsc if denom_dsc else 0.0
            iou = tp/denom_iou   if denom_iou else 0.0
            hd  = hd95(pred, gt)

            dscs.append(dsc)
            ious.append(iou)
            hd95s.append(hd)

            identifier = original_name
            if args.vis == 'composite':
                save_imgs(
                    img.cpu(), seg_bin.cpu(), prob.cpu(),
                    identifier, pred_dir, ds_name, 0.5
                )
            elif args.vis == 'separate':
                save_all_individually(
                    img.cpu(), seg_bin.cpu(), prob.cpu(),
                    identifier, pred_dir, ds_name, 0.5
                )
                Image.fromarray(pred * 255).save(
                    os.path.join(pred_dir, f"{identifier}.png")
                )
            elif args.vis == 'pred':
                save_predicted_mask(
                    prob.cpu(), identifier, pred_dir, ds_name, 0.5
                )
                Image.fromarray(pred * 255).save(
                    os.path.join(pred_dir, f"{identifier}.png")
                )

    df = pd.DataFrame({
        'Name': names + ['Avg', 'Std'],
        'DSC' : dscs  + [np.mean(dscs),  np.std(dscs,  ddof=1)],
        'IoU' : ious  + [np.mean(ious),  np.std(ious,  ddof=1)],
        'HD95': hd95s + [np.mean(hd95s), np.nan],   
    })
    df.to_csv(os.path.join(cfg['snapshot_path'], 'results.csv'), index=False)

    mean_dsc, std_dsc = float(df['DSC'].iloc[-2]),  float(df['DSC'].iloc[-1])
    mean_iou, std_iou = float(df['IoU'].iloc[-2]),  float(df['IoU'].iloc[-1])
    mean_hd           = float(df['HD95'].iloc[-2])  

    logging.info("@@ Test finished in %d s", int(time.time() - t0))
    logging.info("==============  Summary  ==============")
    logging.info("DSC      : %.4f ± %.4f", mean_dsc,  std_dsc)
    logging.info("IoU      : %.4f ± %.4f", mean_iou,  std_iou)
    logging.info("HD95 (px): %.2f",        mean_hd)
    logging.info("=======================================")
    logging.info("Visual outputs saved to %s", pred_dir)

if __name__ == '__main__': 
    main()
