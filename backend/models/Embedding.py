import os
import torch
from torch import nn
from models.Net import Net
from utils.bicubic import BicubicDownSample
from datasets.image_dataset import ImagesDataset


class Embedding(nn.Module):
    def __init__(self, opts):
        super(Embedding, self).__init__()  
        self.opts = opts
        self.net = Net(self.opts)
        self.load_downsampling()

    def load_downsampling(self):
        factor = self.opts.size // 256
        self.downsample = BicubicDownSample(factor=factor)

    def invert_images_in_W(self, image_name=str):
        tqdm = tqdm(range(self.opts.W_steps), desc="W+ Inversion", leave=False)

        img_path = os.path.join(self.opts.input_dir, f"{image_name}.png")

        ref_H, ref_L, _ = ImagesDataset(opts=self.opts, image_path=img_path)[0]

        ref_H.to(self.opts.device)
        ref_L.to(self.opts.device)

        latent = []

        if (self.opts.tile_latent):
            tmp = self.net.latent_avg.clone().detach().cuda()
            tmp.requires_grad = True
            for i in range(self.net.layer_num):
                latent.append(tmp)
            optimizer_W = torch.optim.Adam([tmp], lr=self.opts.learning_rate)
        else:
            for i in range(self.net.layer_num):
                tmp = self.net.latent_avg.clone().detach().cuda()
                tmp.requires_grad = True
                latent.append(tmp)
            optimizer_W = torch.optim.Adam(latent, lr=self.opts.learning_rate)

        
        for step in tqdm:
            optimizer_W.zero_grad()
            latent_in = torch.stack(latent).clone().unsqueeze(0)

            loss_dict = {}

            image, _ = self.net.generator([latent_in], input_is_latent=True, return_latents=False)
