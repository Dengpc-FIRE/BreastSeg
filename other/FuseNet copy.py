"""
FuseNet script converted from FuseNet.ipynb

Run: python FuseNet.py --input_path ./input_images/ --output_path ./output/

This script trains/evaluates the FuseNet self-supervised segmentation demo on images
placed in `input_images/image` and optional GT in `input_images/GT`.
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as T

import cv2
import sys
import os
import numpy as np
import random
import glob
from matplotlib import pyplot as plt

from utils import read_image, dice_metric, xor_metric, hm_metric, create_mask, cross_entropy
from model_utils import Encoder, ProjectionHead, MixFFN_skip, CrossAttentionBlock

from einops import rearrange
from einops.layers.torch import Rearrange


class Model(nn.Module):
    """
    Model class copied from the notebook. See notebook for detailed comments.
    """
    def __init__(self, input_dim, image_embed, augmented_embed, input_size=(256, 256),
                 temperature=5.0, dropout=0.1, beta=16, alpha=3):
        super(Model, self).__init__()

        input_H, input_W = input_size
        self.H = input_H

        self.beta = 16  # Downsampling factor
        self.alpha = 3  # Main path scaling factor
        self.img_enc = Encoder(input_dim, image_embed)
        self.aug_enc = Encoder(input_dim, image_embed)

        self.image_projection = ProjectionHead(embedding_dim=image_embed, projection_dim=image_embed, dropout=dropout)
        self.aug_projection = ProjectionHead(embedding_dim=augmented_embed, projection_dim=augmented_embed, dropout=dropout)
        self.temperature = temperature

        self.cross_attn = CrossAttentionBlock(in_channels=image_embed, key_channels=image_embed,
                                              value_channels=image_embed, height=input_H, width=input_W)

        self.patch_size = self.H//8 #32
        self.dim = image_embed
        patch_dim = self.dim * self.patch_size * self.patch_size

        self.to_patch_embedding_img = nn.Sequential(
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = self.patch_size, p2 = self.patch_size),
            nn.Linear(patch_dim, self.dim))

        self.to_patch_embedding_aug = nn.Sequential(
            Rearrange('b (h p1) (w p2) c -> b (h w) (p1 p2 c)', p1 = self.patch_size, p2 = self.patch_size),
            nn.Linear(patch_dim, self.dim))    

        self.bn1 = nn.BatchNorm2d(image_embed)
        self.bn2 = nn.BatchNorm2d(image_embed)

    def forward(self, x, augmented_xs):

        # extract feature representations of the main image
        img_f = self.img_enc(x)
        img_f = rearrange(img_f, 'b c h w -> b (h w) c')

        # support single tensor (old behavior) or list/tuple of augmentations
        aug_fs = []
        if isinstance(augmented_xs, (list, tuple)):
            for aug in augmented_xs:
                af = self.aug_enc(aug)
                af = rearrange(af, 'b c h w -> b (h w) c')
                aug_fs.append(af)
        else:
            af = self.aug_enc(augmented_xs)
            af = rearrange(af, 'b c h w -> b (h w) c')
            aug_fs.append(af)

        # Getting Image and augmented image Embeddings (with same dimension)
        img_e = self.image_projection(img_f)
        aug_es = [self.aug_projection(af) for af in aug_fs]

        # Calculating CLIP: compute patch embeddings, normalize, and average over augmentations
        img_e_r = self.bn1(rearrange(img_e, 'b (h w) c -> b c h w', h=self.H)).permute(0, 2, 3, 1)
        img_e_patch = self.to_patch_embedding_img(img_e_r)
        img_e_norm = img_e_patch / img_e_patch.norm(dim=-1, keepdim=True)

        aug_patches = []
        for aug_e in aug_es:
            aug_e_r = self.bn2(rearrange(aug_e, 'b (h w) c -> b c h w', h=self.H)).permute(0, 2, 3, 1)
            aug_e_patch = self.to_patch_embedding_aug(aug_e_r)
            aug_e_norm = aug_e_patch / (aug_e_patch.norm(dim=-1, keepdim=True) + 1e-8)
            aug_patches.append(aug_e_norm)

        # average normalized patch embeddings across augmentations
        aug_e_norm_avg = torch.stack(aug_patches, dim=0).mean(dim=0)

        clip_sim = (img_e_norm @ aug_e_norm_avg.mT) / self.temperature
        img_e_sim = img_e_norm @ img_e_norm.mT
        aug_e_sim = aug_e_norm_avg @ aug_e_norm_avg.mT
        clip_targets = F.softmax((img_e_sim + aug_e_sim) / 2 * self.temperature, dim=-1)

        # Cross attention: compute attention with each augmented embedding and sum their contributions
        attn = None
        for aug_e in aug_es:
            attn_1 = self.cross_attn(img_e * self.alpha, aug_e * 0.8)
            attn_2 = self.cross_attn(aug_e * 0.8, img_e * self.alpha)
            if attn is None:
                attn = attn_1 + attn_2
            else:
                attn = attn + (attn_1 + attn_2)

        _, edge1 = torch.max(attn, 1)
        attn_down = torchvision.transforms.functional.resize(attn, 256//self.beta, antialias=True)
        attn_up = torchvision.transforms.functional.resize(attn_down, 256, antialias=True)
        _, edge2 = torch.max(attn_up, 1)
        edge = edge1 - edge2

        return edge, attn, clip_sim, clip_targets


def build_parser():
    parser = argparse.ArgumentParser(description='FuseNet: Self-Supervised Dual-Path Network for Medical Image Segmentation')
    parser.add_argument('--nChannel', metavar='N', default=64, type=int, 
                        help='number of channels')
    parser.add_argument('--maxIter', metavar='T', default=50, type=int, 
                        help='number of maximum iterations')
    parser.add_argument('--minLabels', metavar='minL', default=3, type=int, 
                        help='minimum number of labels')
    parser.add_argument('--lr', metavar='LR', default=0.005, type=float, 
                        help='learning rate')

    parser.add_argument('--input_path', metavar='INPUT', default='./input_images/', 
                        help='input image folder path')
    parser.add_argument('--save_output', metavar='SAVE', default=True, 
                        help='whether to save output ot not')
    parser.add_argument('--output_path', metavar='OUTPUT', default='./output/', 
                        help='output folder path')

    parser.add_argument('--loss_ce_coef', metavar='CE', default=2.5, type=float, 
                        help='Cross entropy loss weighting factor')
    parser.add_argument('--loss_clip_coef', metavar='AT', default=0.5, type=float, 
                        help='Clip loss weighting factor')
    parser.add_argument('--loss_b_coef', metavar='Spatial', default=0.5, type=float, 
                        help='Boundary loss weighting factor')

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.save_output:
        SAVE_PATH = args.output_path
        os.makedirs(SAVE_PATH, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Loading Data
    IMG_PATH = args.input_path
    image_dir = os.path.join(IMG_PATH, 'image')
    gt_dir = os.path.join(IMG_PATH, 'GT')
    vibrant_files = sorted(glob.glob(os.path.join(image_dir, '*_VIBRANT.*')))

    img_size = 256

    for img_num, vibrant_file in enumerate(vibrant_files):
        ##### Read VIBRANT (initial image) #####
        image = read_image(vibrant_file, img_size).to(device)

        # collect SUB files for this case
        case_name = os.path.basename(vibrant_file).split('_VIBRANT')[0]
        sub_pattern = os.path.join(image_dir, f"{case_name}_SUB*.*")
        sub_files = sorted(glob.glob(sub_pattern))
        if len(sub_files) == 0:
            print(f"[WARN] no SUB files found for {case_name}, skipping")
            continue
        aug_imgs = [read_image(p, img_size).to(device) for p in sub_files]

        ##### Load Model #####
        model = Model(input_dim=3, image_embed=64, augmented_embed=64,
                      input_size=(img_size, img_size), temperature=5.0, dropout=0.1,
                      beta=16, alpha=3).to(device)
        model.train()

        ##### Settings #####
        zero_img = torch.zeros(image.shape[2], image.shape[3]).to(device)
        
        loss_ce = torch.nn.CrossEntropyLoss()
        loss_s = torch.nn.L1Loss()
        
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        label_colours = np.random.randint(255, size=(128, 3))

        # optional augmentation on SUB images can be applied here if desired
        # currently we use raw SUB images as provided
        aug_img_list = aug_imgs

        ##### Training #####
        for batch_idx in range(args.maxIter):

            optimizer.zero_grad()
            edge, output, clip_logits, clip_targets = model(image, aug_img_list)
            
            ### Output
            output, clip_logits, clip_targets = output[0], clip_logits[0], clip_targets[0]        
            output = output.permute(1, 2, 0).contiguous().view(-1, args.nChannel*2)
                    
            _, target = torch.max(output, 1)
            img_target = target.data.cpu().numpy()
            img_target_rgb = np.array([label_colours[c % args.nChannel] for c in img_target])
            img_target_rgb = img_target_rgb.reshape(image.shape[2], image.shape[3], image.shape[1]).astype(np.uint8)
            
            ### Cross-entropy loss function         
            loss_ce_value = args.loss_ce_coef * loss_ce(output, target)
            
            ### Boundary Loss
            loss_edge = args.loss_b_coef * loss_s(edge[0], zero_img)  
            
            ### CLIP loss 
            aug_loss = cross_entropy(clip_logits, clip_targets, 'mean')
            img_loss = cross_entropy(clip_logits.T, clip_targets.T, 'mean')
            loss_clip = args.loss_clip_coef * ((img_loss + aug_loss) / 2.0)
            
            ### Optimization        
            loss = loss_ce_value + loss_clip + loss_edge
            loss.backward()
            optimizer.step()
            
            nLabels = len(np.unique(img_target))
            print(batch_idx, '/', args.maxIter, '|', ' label num:', nLabels, ' | loss:', round(loss.item(), 4),
                    '| CE:', round(loss_ce_value.item(), 4), '| CLIP:', round(loss_clip.item(), 4),
                    '| B:', round(loss_edge.item(), 4))
                
            if nLabels <= args.minLabels and batch_idx>=5:
                print (f"Number of labels have reached {nLabels}")
                break

        ##### Evaluate #####
        edge, output, _, _ = model(image, aug_img_list)
        output = output[0].permute(1, 2, 0).contiguous().view(-1, args.nChannel*2)
        _, target = torch.max(output, 1)
        img_target = target.data.cpu().numpy()
        img_eval_output = np.array([label_colours[c % args.nChannel] for c in img_target])
        img_eval_output = img_eval_output.reshape(image.shape[2], image.shape[3], image.shape[1]).astype(np.uint8)

        ##### Visualization #####
        fig, axes = plt.subplots(1, 4, figsize=(8, 8))
        axes[0].imshow(img_eval_output)
        axes[1].imshow(image[0].permute(1, 2, 0).cpu().detach().numpy()[..., ::-1])
        # show the first SUB as an example
        axes[2].imshow(aug_img_list[0][0].permute(1, 2, 0).cpu().detach().numpy()[...,::-1])
        axes[3].imshow(edge[0].cpu().detach().numpy())
        axes[0].set_title('Prediction')
        axes[1].set_title('Input Image')
        axes[2].set_title('Augmented Image')
        axes[3].set_title('Edge SR')    
        axes[0].axis('off')
        axes[1].axis('off')
        axes[2].axis('off')
        axes[3].axis('off')
        plt.show()

        if args.save_output:
            name = os.path.basename(vibrant_file).split('.')[0]
            cv2.imwrite(SAVE_PATH + '/FuseNet_mask_' + name + '.png', img_eval_output)
            cv2.imwrite(SAVE_PATH + '/FuseNet_img_' + name + '.png', image[0].permute(1, 2, 0).cpu().detach().numpy()*255)
            # save first SUB as example
            cv2.imwrite(SAVE_PATH + '/FuseNet_aug_' + name + '.png', aug_img_list[0][0].permute(1, 2, 0).cpu().detach().numpy()*255)

        print('-------------------------------', '\n')


if __name__ == '__main__':
    main()
