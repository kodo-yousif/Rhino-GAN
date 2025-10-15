import torch
import torchvision
import numpy as np
from torch.nn import functional as F
toPIL = torchvision.transforms.ToPILImage()

HAIR_REGION = 10
NOSE_REGION = 2
SKIN_REGION = 1
IGNORE_CODE = 255
nose_idxs = torch.arange(27, 36, device="cuda")

COLOR_MAP = np.array([
    [0, 0, 0],         # 0: background - black
    [255, 0, 0],       # 1: red
    [0, 255, 0],       # 2: green
    [0, 0, 255],       # 3: blue
    [255, 255, 0],     # 4: yellow
    [255, 0, 255],     # 5: magenta
    [0, 255, 255],     # 6: cyan
    [255, 128, 0],     # 7: orange
    [128, 0, 255],     # 8: violet
    [0, 128, 255],     # 9: sky blue
    [128, 255, 0],     # 10: lime
    [255, 0, 128],     # 11: pink-red
    [0, 255, 128],     # 12: aqua green
    [128, 128, 0],     # 13: olive
    [0, 128, 128],     # 14: teal
    [192, 192, 192],   # 15: light gray
], dtype=np.uint8)


def verbose(ctx, title, dict, loss, pbar):
    if ctx.opts.verbose:
        desc_items = [f"{k}: {v:.3f}" for k, v in dict.items()]
        pbar.set_description(f"{title}: {loss:.3f}, " + ", ".join(desc_items))


def compare_LPIPS(ctx, ref_L, gen_L, foreign_mask= None):
    if foreign_mask is not None :
        mask = F.interpolate(foreign_mask.unsqueeze(1), size=(256, 256), mode='nearest')
        lpips_loss = ctx.lpips(gen_L * mask, ref_L * mask).sum() * ctx.opts.percept_lambda
    else:
        lpips_loss = ctx.lpips(gen_L, ref_L).sum() * ctx.opts.percept_lambda

    return lpips_loss


def save_as_image(latent, path):
    save_im = toPIL(((latent[0] + 1) / 2).detach().cpu().clamp(0, 1))
    save_im.save(path)