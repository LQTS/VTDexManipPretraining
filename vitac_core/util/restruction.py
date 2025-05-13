import json
import cv2
import torch
from pathlib import Path
import os
from einops import rearrange
import matplotlib.pyplot as plt
import numpy as np
def save_recons_tac(
    ori_tacs: torch.Tensor,
    recons_tacs: torch.Tensor,
    save_dir_input: Path,
    identify: str
) -> None:


    save_dir = save_dir_input / identify
    os.makedirs(str(save_dir), exist_ok=True)

    result_dict = dict()
    for bid in range(ori_tacs.shape[0]):
        ori_tac = ori_tacs[bid]
        recon_tac = recons_tacs[bid]
        # to Numpy format
        ori_tac, recon_tac = ori_tac.cpu().detach().numpy(), recon_tac.cpu().detach().numpy()
        # save to Dict
        result_dict[bid] = {
            "origin": ori_tac.tolist(),
            "recons": recon_tac.tolist(),
            "delta_": (recon_tac - ori_tac).tolist()
        }
    # save as json file
    json_str = json.dumps(result_dict, indent=2)
    save_json_path = save_dir / "res.json"
    with open(str(save_json_path), 'w') as f:
        f.write(json_str)

def save_recons_imgs(
    ori_imgs: torch.Tensor,
    recons_imgs: torch.Tensor,
    save_dir_input: Path,
    identify: str,
    online_normalization,
    mask_imgs: torch.Tensor=None,
) -> None:

    # de online transforms function
    def de_online_transform(ori_img, recon_img, norm=online_normalization):
        # rearrange
        ori_imgs = rearrange(ori_img,"c h w -> h w c")
        recon_imgs = rearrange(recon_img, "c h w -> h w c")
        # to Numpy format
        ori_imgs = ori_imgs.detach().numpy()
        recon_imgs = recon_imgs.detach().numpy()
        # # DeNormalize
        ori_imgs = np.array(norm[0]) + ori_imgs * np.array(norm[1])
        recon_imgs = np.array(norm[0]) + recon_imgs * np.array(norm[1])
        # to cv format
        ori_imgs = np.uint8(ori_imgs * 255)
        recon_imgs = np.uint8(recon_imgs * 255)

        return ori_imgs, recon_imgs
    def save_masked_recons_area(recon_img, mask_img, save_path):
        mask = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 125, 255, cv2.THRESH_BINARY_INV)
        recon_img_ = cv2.bitwise_and(recon_img, recon_img, mask=mask)
        cv2.imwrite(save_path, recon_img_)
    def save_visible_area(recon_img, mask_img, save_path):
        mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask_img, 125, 255, cv2.THRESH_BINARY)
        recon_img_ = cv2.bitwise_and(recon_img, recon_img, mask=mask)
        cv2.imwrite(save_path, recon_img_)

    save_dir = save_dir_input / identify
    os.makedirs(str(save_dir), exist_ok=True)

    for bid in range(ori_imgs.shape[0]):
        ori_img = ori_imgs[bid]
        recon_img = recons_imgs[bid]
        mask_img = None
        if mask_imgs is not None:
            mask_img = mask_imgs[bid]
            mask_img = rearrange(mask_img, "c h w -> h w c")
            mask_img = mask_img.detach().numpy()
            mask_img = np.uint8(mask_img * 255)

            mask_save_path = save_dir / f"{bid}_mask.jpg"
            cv2.imwrite(str(mask_save_path), mask_img)
        # de online norm
        ori_img, recon_img = de_online_transform(ori_img, recon_img)

        ori_save_path = save_dir / f"{bid}_raw.jpg"
        recon_save_path = save_dir / f"{bid}_recon.jpg"
        masked_recon_save_path = save_dir / f"{bid}_masked_recon.jpg"
        visible_recon_save_path = save_dir / f"{bid}_visible_recon.jpg"

        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_RGB2BGR)
        recon_img = cv2.cvtColor(recon_img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(str(ori_save_path), ori_img)
        cv2.imwrite(str(recon_save_path), recon_img)
        if mask_imgs is not None:
            save_masked_recons_area(recon_img, mask_img, str(masked_recon_save_path))
            save_visible_area(recon_img, mask_img, str(visible_recon_save_path))


def show_image(image, title='', imagenet_mean=np.array([0.485, 0.456, 0.406]),
               imagenet_std=np.array([0.229, 0.224, 0.225])):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


def visualize_recon_image(model, save_dir, img, img_recon, mask):
    # x = torch.tensor(img)
    #
    # # make it a batch-like
    # x = x.unsqueeze(dim=0)
    # x = torch.einsum('nhwc->nchw', x)
    #
    # # run MAE
    # loss, y, mask = model(x.float(), mask_ratio=0.75)
    if model.norm_pixel_loss:
        targets = model.patchify(img)
        # Normalize targets parameters
        mu, var = targets.mean(dim=-1, keepdim=True), targets.var(dim=-1, unbiased=True, keepdim=True)
        # targets = (targets - mu) / ((var + 1e-6) ** 0.5)

        # de normalize patched_imgs
        img_recon = img_recon * ((var + 1e-6) ** 0.5) + mu

    y = model.depatchify(img_recon)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()


    # visualize the mask
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_size ** 2 * 3)  # (N, H*W, p*p*3)
    mask = model.depatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()

    x = torch.einsum('nchw->nhwc', img)

    # masked image
    im_masked = x * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x * (1 - mask) + y * mask

    os.makedirs(save_dir, exist_ok=True)

    for i in range(min(im_paste.shape[0], 10)):
        # make the plt figure larger
        plt.rcParams['figure.figsize'] = [20, 6]

        plt.subplot(1, 4, 1)
        show_image(x[i], "original")

        plt.subplot(1, 4, 2)
        show_image(im_masked[i], "masked")

        plt.subplot(1, 4, 3)
        show_image(y[i], "reconstruction")

        plt.subplot(1, 4, 4)
        show_image(im_paste[i], "reconstruction + visible")

        plt.savefig(str(save_dir) + f'/recon_pic_{i}.png')
        # plt.show()

    return dict(original=x, masked=im_masked, reconstruction=y, re_vis=im_paste)