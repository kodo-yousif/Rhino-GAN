import os
import torch
import numpy as np
from torch import nn
from pathlib import Path
from models.Net import Net
from pydantic import BaseModel
from typing import Optional, List
from utils.bicubic import BicubicDownSample
from utils.model_utils import download_weight
from models.face_parsing.model import BiSeNet
from models.StructureTransfer import StructureTransfer
from utils.helpers import save_as_image


class FineTuneContext(BaseModel):
    ref_path: str
    inversion_name: str
    nose_path: Optional[str] = None
    landmarks: Optional[List[List[float]]] = None
    segmentation: Optional[List[List[float]]] = None


class Tunning(nn.Module):
    def __init__(self, opts):
        super(Tunning, self).__init__()
        self.opts = opts
        self.net = Net(self.opts)
        self.load_downsampling()
        self.load_loss_functions()

    def load_downsampling(self):
        self.downsample = BicubicDownSample(factor=self.opts.size // 512)
        self.downsample_256 = BicubicDownSample(factor=self.opts.size // 256)

    def load_loss_functions(self):
        self.seg = BiSeNet(n_classes=16)
        self.seg.to(self.opts.device)
        if not os.path.exists(self.opts.seg_ckpt):
            download_weight(self.opts.seg_ckpt)
        self.seg.load_state_dict(torch.load(self.opts.seg_ckpt))
        for param in self.seg.parameters():
            param.requires_grad = False
        self.seg.eval()

        self.transferModel = StructureTransfer(self.opts)

    def save_result(self,F,S, ref_path):
            gen_im, _ = self.net.generator([S], input_is_latent=True, return_latents=False, start_layer=4, end_layer=8, layer_in=F)

            output_folder_path = os.path.join(Path(ref_path).parent, "tuned")

            os.makedirs(output_folder_path, exist_ok=True)

            FS_path = os.path.join(output_folder_path, "FS.npz")
            image_result_path = os.path.join(output_folder_path, f"FS.png")

            np.savez(
                FS_path,
                latent_in=S.detach().cpu().numpy(),
                latent_F=F.detach().cpu().numpy()
            )

            save_as_image(gen_im, image_result_path)

    def nose_transfer(self, item: FineTuneContext):
        print(item.nose_path)
        print(item.ref_path)
        if item.nose_path == "self" or item.ref_path == item.nose_path:
            print("loadinggggggggggg")
        else:
            S, F = self.transferModel.transfer(item.ref_path, item.nose_path, item.inversion_name)
            self.save_result(F, S, item.ref_path)
            return

    def tune_image(self, item: FineTuneContext):

        if item.nose_path is not None:
            self.nose_transfer(item)

        else:
            if item.segmentation is not None:
                item.segmentation = torch.tensor(item.segmentation, dtype=torch.long, device= self.opts.device).unsqueeze(0)

            # self.tune_using_f(item)