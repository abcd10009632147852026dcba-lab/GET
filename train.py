# train.py

import os, time, re, yaml, logging, argparse
from copy import deepcopy
import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import OmegaConf
from monai.losses import DiceLoss
from tensorboardX import SummaryWriter
from tqdm import tqdm
from monai.losses import TverskyLoss

from utils.image_dataset            import Image_Dataset
from utils.tools                   import seed_reproducer, save_checkpoint
from utils.get_logger              import open_log
from networks.embedding_translation import MainPipeline
from networks.vae_setup.autoencoder   import AutoencoderKL
from networks.vae_setup.distributions import DiagonalGaussianDistribution

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32       = True
AMP = torch.cuda.is_available()

# focal‑tversky loss 
class FocalTversky(torch.nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, gamma=0.75, eps=1e-6):
        super().__init__(); self.a,self.b,self.g,self.eps=alpha,beta,gamma,eps
    def forward(self, p, t):
        tp = (p * t).sum((1,2,3))
        fp = ((1 - t) * p).sum((1,2,3))
        fn = (t * (1 - p)).sum((1,2,3))
        tv = (tp + self.eps) / (tp + self.a * fp + self.b * fn + self.eps)
        return torch.pow(1 - tv, self.g).mean()

# helpers 
def get_mu_sigma(post, scale):
    mu, logvar = post.mu_and_sigma()
    return scale * mu, logvar

def vae_decode(vae, mu_pred, scale):
    z   = mu_pred.div(scale).clamp(-10, 10)
    seg = vae.decode(z).mean(1, True).clamp(-1, 1)
    seg = torch.nan_to_num(seg, nan=0.0, posinf=1.0, neginf=-1.0)
    return ((seg + 1) * 0.5).clamp(0, 1)

def kl_iso(mu1, mu2):                       # isotropic KL between μ vectors
    return 0.5 * torch.sum((mu1 - mu2) ** 2, dim=1).mean()

def parse_cfg():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='./configs/train.yaml')
    return p.parse_args()

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1 - self.decay)
    def copy_to(self, model):
        model.load_state_dict(self.shadow, strict=True)

def run_trainer():
    args = parse_cfg()
    cfg  = yaml.safe_load(open(args.config))

    if isinstance(cfg.get('save_freq'), str):
        cfg['save_freq'] = int(re.findall(r'\d+', cfg['save_freq'])[0])

    # folders
    ts = time.strftime('%Y%m%d%H%M', time.localtime())
    cfg['snapshot_path'] = os.path.join(cfg['snapshot_path'], ts)
    cfg['log_path']      = os.path.join(cfg['snapshot_path'], 'logs')
    os.makedirs(cfg['snapshot_path'], exist_ok=True)
    os.makedirs(cfg['log_path'],      exist_ok=True)

    # env
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, cfg['GPUs']))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_reproducer(cfg['seed'])

    # logging
    open_log(args, cfg)
    logging.info("Config:\n%s", cfg); 
    writer = SummaryWriter(cfg['log_path'])

    # datasets
    train_loader = DataLoader(
        Image_Dataset(cfg['pickle_file_path'], 'train'),
        batch_size=cfg['batch_size'], shuffle=True, drop_last=True,
        pin_memory=True, num_workers=cfg['num_workers'],
        persistent_workers=True, prefetch_factor=2
    )
    valid_loader = DataLoader(
        Image_Dataset(cfg['pickle_file_path'], 'test'),
        batch_size=cfg['batch_size'], shuffle=False, drop_last=False,
        pin_memory=True, num_workers=cfg['num_workers'],
        persistent_workers=True, prefetch_factor=2
    )

    # backbone
    mapper = MainPipeline(
        in_channel   = cfg['in_channel'],
        out_channels = cfg['out_channels'],
        ch           = cfg['ch'],
        ch_mult      = tuple(cfg['ch_mult']),
        drop_rate    = 0.1
    ).to(device)

    # VAE (frozen)
    vae_cfg = OmegaConf.load('./configs/sd-first-stage-VAE.yaml')
    vae = AutoencoderKL(**vae_cfg.first_stage_config.params)
    vae.load_state_dict(
        torch.load('SD-VAE-weight/768-v-ema-first-stage-VAE.ckpt',
                   map_location='cpu')['state_dict'],
        strict=True)
    vae.eval().requires_grad_(False).to(device)
    scale = vae_cfg.first_stage_config.scale_factor

    # opt & sched
    opt = torch.optim.AdamW(mapper.parameters(), lr=cfg['lr'], weight_decay=1e-4)
    warm = cfg.get('warmup_epochs', 5)
    cosine = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=cfg.get('T_restart', 150), T_mult=2)
    sch = torch.optim.lr_scheduler.SequentialLR(
        opt,
        [torch.optim.lr_scheduler.LinearLR(opt, 1e-6, 1.0, warm), cosine],
        milestones=[warm]
    )

    # DS weights (out, level1, level2, level3)
    ds_w = torch.tensor([1.0, 0.4, 0.3, 0.2], device=device)

    # losses
    l1     = torch.nn.L1Loss()
    diceL  = DiceLoss()
    ftvL   = FocalTversky()
    scaler = torch.amp.GradScaler(enabled=AMP)
    ema    = EMA(mapper, decay=0.999)

    best_dice, best_ep = -1, 0
    ds_keys = ['out', 'level1', 'level2', 'level3']  # same order as ds_w

    # train
    for epoch in range(1, cfg['epochs'] + 1):
        t0 = time.time(); mapper.train()
        meters = {k: [] for k in ('tot', 'rec', 'dice', 'ftv', 'kl')}

        for batch in tqdm(train_loader, desc=f"Train {epoch}"):
            img = (2 * batch['img'] - 1).to(device)
            seg = (batch['seg'].permute(0, 3, 1, 2) / 255.).to(device)
            seg_bin = seg.mean(1, True)

            with torch.no_grad():
                mu_img, log_img = get_mu_sigma(vae.encode(img), scale)
                mu_seg,_        = get_mu_sigma(vae.encode(2 * seg - 1), scale)

            if np.random.rand() > 0.5:
                mu_img += (0.5 * log_img.clamp(-10, 10)).exp() * torch.randn_like(mu_img)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda', enabled=AMP):
                outs = mapper(mu_img)

                rec_losses = []
                for wi, k in enumerate(ds_keys):
                    target = mu_seg
                    if outs[k].shape[2:] != mu_seg.shape[2:]:
                        target = F.interpolate(mu_seg, size=outs[k].shape[2:],
                                               mode='bilinear')
                    rec_losses.append(ds_w[wi] * l1(outs[k], target))
                loss_rec = sum(rec_losses)

                prob_main = vae_decode(vae, outs['out'], scale)
                loss_dice = cfg['w_dice'] * diceL(prob_main, seg_bin)
                loss_ftv  = cfg['w_ftv'] * ftvL(prob_main, seg_bin)

                prob_aux = vae_decode(vae, outs['level1'], scale)
                loss_dice += 0.5 * cfg['w_dice'] * diceL(prob_aux, seg_bin)
                loss_ftv  += 0.5 * ftvL(prob_aux, seg_bin)

                loss_kl = cfg['w_kl'] * kl_iso(outs['out'], mu_seg)
                loss = cfg['w_rec'] * loss_rec + loss_dice + loss_ftv + loss_kl  

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            ema.update(mapper)

            # losses
            meters['tot'].append(loss.item())
            meters['rec'].append(loss_rec.item())
            meters['dice'].append(loss_dice.item())
            meters['ftv'].append(loss_ftv.item())
            meters['kl'].append(loss_kl.item())

        sch.step()
        logging.info(
            "Train: L=%.4f rec=%.4f dice=%.4f ftv=%.4f kl=%.4f",
            *(np.mean(meters[k]) for k in ('tot', 'rec', 'dice', 'ftv', 'kl'))
        )

        mapper_ema = deepcopy(mapper).to(device)
        ema.copy_to(mapper_ema); mapper_ema.eval()
        dices = []
        with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=AMP):
            for batch in tqdm(valid_loader, desc="Valid"):
                img = (2 * batch['img'] - 1).to(device)
                seg = (batch['seg'].permute(0, 3, 1, 2) / 255.).to(device).mean(1, True)

                mu_img,_ = get_mu_sigma(vae.encode(img), scale)
                prob = vae_decode(vae, mapper_ema(mu_img)['out'], scale)

                inter = (prob * seg).sum((1, 2, 3))
                union = prob.sum((1, 2, 3)) + seg.sum((1, 2, 3))
                dices.append((2 * inter / (union + 1e-6)).mean().item())

        vdice = float(np.mean(dices))
        if vdice > best_dice:
            best_dice, best_ep = vdice, epoch
            save_checkpoint(mapper_ema, 'best_dice_ema.pth', cfg['snapshot_path'])
        logging.info("Valid: Dice=%.4f", vdice)
        writer.add_scalar('valid/dice', vdice, epoch)

        logging.info(
            "Epoch %d/%d (Dice=%.4f, Best=%.4f@%d) – %ds",
            epoch, cfg['epochs'],
            vdice, best_dice, best_ep, int(time.time() - t0),
        )

        if epoch % cfg['save_freq'] == 0:
            save_checkpoint(mapper, f'epoch_{epoch:03d}.pth', cfg['snapshot_path'])

    writer.close()

if __name__ == '__main__':
    run_trainer()
