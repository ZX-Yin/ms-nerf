import os
import glob
import numpy as np
import cv2
import torch
from torchmetrics.functional import peak_signal_noise_ratio as compute_psnr
from torchmetrics.functional import structural_similarity_index_measure as compute_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
import json
from tqdm import tqdm

def load_image(path):
    return cv2.imread(path)[:,:,::-1]

def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))
_ = torch.manual_seed(202208)
compute_lpips = LPIPS()

# prediction path
path = '/mnt/sda/experiments/cvpr23/Mip-NeRF-360/logs_MS-Mip-NeRF-360/Scene*/test_preds'
# ground truth path
gt_path_root = '/mnt/sda/T3/cvpr23/dataset/synthetic_scenes'
basedir = '/'.join(path.split('/')[:-2])
results_list = sorted(glob.glob(path))
results_dict = {}
for scene in results_list:
    scene_name = scene.split('/')[-2]
    img_list = glob.glob(os.path.join(scene, 'color_0*.png'))
    scene_dict  = {}
    psnrs, ssims, lpipss = [], [], []
    print (f'eval {scene_name}')
    for pred_path in tqdm(img_list):
        img_name = pred_path.split('/')[-1][6:9]
        gt_path = os.path.join(gt_path_root, scene_name, f'test/r_{int(img_name)}.png')

        pred = load_image(pred_path)
        gt   = load_image(gt_path)
        pred_lpips = im2tensor(pred)
        gt_lpips   = im2tensor(gt)
        
        pred = torch.from_numpy((pred / 255.).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        gt   = torch.from_numpy((gt / 255.).astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        
        with torch.no_grad():
            psnr = compute_psnr(pred, gt, data_range=1.0).item()
            ssim = compute_ssim(pred, gt, data_range=1.0).item()
            lpips = compute_lpips(pred_lpips, gt_lpips).item()
        img_dict = {'psnr'  : psnr,
                    'ssim'  : ssim,
                    'lpips' : lpips}
        scene_dict[img_name] = img_dict
        psnrs.append(psnr)
        ssims.append(ssim)
        lpipss.append(lpips)
    avg_psnr = np.mean(psnrs)
    avg_ssim = np.mean(ssims)
    avg_lpips = np.mean(lpipss)
    avg_dict = {'avg_psnr'  : avg_psnr,
                'avg_ssim'  : avg_ssim,
                'avg_lpips' : avg_lpips}
    scene_dict[scene_name] = avg_dict
    with open(os.path.join(scene, 'psnr_ssim_lpips.json'), 'w')  as f:
        json.dump(scene_dict, f)
    results_dict[scene_name] = avg_dict

whole_psnrs, whole_ssims, whole_lpipss = [], [], []
for _, val in results_dict.items():
    whole_psnrs.append(val['avg_psnr'])
    whole_ssims.append(val['avg_ssim'])
    whole_lpipss.append(val['avg_lpips'])
whole_avg = {
    'avg_psnr'  : np.mean(whole_psnrs),
    'avg_ssim'  : np.mean(whole_ssims),
    'avg_lpips' : np.mean(whole_lpipss)
}
results_dict['whole'] = whole_avg
with open(os.path.join(basedir, 'psnr_ssim_lpips_syn.json'), 'w') as f:
    json.dump(results_dict, f)