import sys
sys.path.insert(0,'jax/internal/pycolmap')
sys.path.insert(0,'jax/internal/pycolmap/pycolmap')
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

from typing import Mapping, Optional, Sequence, Text, Tuple, Union
import enum
import types

import pycolmap

_Array = Union[np.ndarray]

class ProjectionType(enum.Enum):
  """Camera projection type (standard perspective pinhole or fisheye model)."""
  PERSPECTIVE = 'perspective'
  FISHEYE = 'fisheye'
  
def intrinsic_matrix(fx: float,
                     fy: float,
                     cx: float,
                     cy: float,
                     xnp: types.ModuleType = np) -> _Array:
  """Intrinsic matrix for a pinhole camera in OpenCV coordinate system."""
  return xnp.array([
      [fx, 0, cx],
      [0, fy, cy],
      [0, 0, 1.],
  ])
  
def file_exists(pth):
  return os.path.exists(pth)
  
def listdir(pth):
  return os.listdir(pth)



class NeRFSceneManager(pycolmap.SceneManager):
  """COLMAP pose loader.

  Minor NeRF-specific extension to the third_party Python COLMAP loader:
  google3/third_party/py/pycolmap/scene_manager.py
  """

  def process(
      self
  ) -> Tuple[Sequence[Text], np.ndarray, np.ndarray, Optional[Mapping[
      Text, float]], ProjectionType]:
    """Applies NeRF-specific postprocessing to the loaded pose data.

    Returns:
      a tuple [image_names, poses, pixtocam, distortion_params].
      image_names:  contains the only the basename of the images.
      poses: [N, 4, 4] array containing the camera to world matrices.
      pixtocam: [N, 3, 3] array containing the camera to pixel space matrices.
      distortion_params: mapping of distortion param name to distortion
        parameters. Cameras share intrinsics. Valid keys are k1, k2, p1 and p2.
    """

    self.load_cameras()
    self.load_images()
    # self.load_points3D()  # For now, we do not need the point cloud data.

    # Assume shared intrinsics between all cameras.
    cam = self.cameras[1]

    # Extract focal lengths and principal point parameters.
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    pixtocam = np.linalg.inv(intrinsic_matrix(fx, fy, cx, cy))

    # Extract extrinsic matrices in world-to-camera format.
    imdata = self.images
    w2c_mats = []
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    for k in imdata:
      im = imdata[k]
      rot = im.R()
      trans = im.tvec.reshape(3, 1)
      w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
      w2c_mats.append(w2c)
    w2c_mats = np.stack(w2c_mats, axis=0)

    # Convert extrinsics to camera-to-world.
    c2w_mats = np.linalg.inv(w2c_mats)
    poses = c2w_mats[:, :3, :4]

    # Image names from COLMAP. No need for permuting the poses according to
    # image names anymore.
    names = [imdata[k].name for k in imdata]

    # Switch from COLMAP (right, down, fwd) to NeRF (right, up, back) frame.
    poses = poses @ np.diag([1, -1, -1, 1])

    # Get distortion parameters.
    type_ = cam.camera_type

    if type_ == 0 or type_ == 'SIMPLE_PINHOLE':
      params = None
      camtype = ProjectionType.PERSPECTIVE

    elif type_ == 1 or type_ == 'PINHOLE':
      params = None
      camtype = ProjectionType.PERSPECTIVE

    if type_ == 2 or type_ == 'SIMPLE_RADIAL':
      params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
      params['k1'] = cam.k1
      camtype = ProjectionType.PERSPECTIVE

    elif type_ == 3 or type_ == 'RADIAL':
      params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
      params['k1'] = cam.k1
      params['k2'] = cam.k2
      camtype = ProjectionType.PERSPECTIVE

    elif type_ == 4 or type_ == 'OPENCV':
      params = {k: 0. for k in ['k1', 'k2', 'k3', 'p1', 'p2']}
      params['k1'] = cam.k1
      params['k2'] = cam.k2
      params['p1'] = cam.p1
      params['p2'] = cam.p2
      camtype = ProjectionType.PERSPECTIVE

    elif type_ == 5 or type_ == 'OPENCV_FISHEYE':
      params = {k: 0. for k in ['k1', 'k2', 'k3', 'k4']}
      params['k1'] = cam.k1
      params['k2'] = cam.k2
      params['k3'] = cam.k3
      params['k4'] = cam.k4
      camtype = ProjectionType.FISHEYE

    return names, poses, pixtocam, params, camtype

def load_gts(data_dir, llffhold = 8, factor = 8, load_alphabetical = True):
    image_dir_suffix = ''
    # Use downsampling factor (unless loading training split for raw dataset,
    # we train raw at full resolution because of the Bayer mosaic pattern).
    if factor > 0:
      image_dir_suffix = f'_{factor}'

    # Copy COLMAP data to local disk for faster loading.
    colmap_dir = os.path.join(data_dir, 'sparse/0/')

    # Load poses.
    if os.path.exists(colmap_dir):
      pose_data = NeRFSceneManager(colmap_dir).process()
    else:
      raise ValueError(f'Image folder {colmap_dir} does not exist.')
    image_names = pose_data[0]

    # Previous NeRF results were generated with images sorted by filename,
    # use this flag to ensure metrics are reported on the same test set.
    if load_alphabetical:
      inds = np.argsort(image_names)
      image_names = [image_names[i] for i in inds]
      
    colmap_image_dir = os.path.join(data_dir, 'images')
    image_dir = os.path.join(data_dir, 'images' + image_dir_suffix)
    for d in [image_dir, colmap_image_dir]:
        if not file_exists(d):
            raise ValueError(f'Image folder {d} does not exist.')
    # Downsampled images may have different names vs images used for COLMAP,
    # so we need to map between the two sorted lists of files.
    colmap_files = sorted(listdir(colmap_image_dir))
    image_files = sorted(listdir(image_dir))
    colmap_to_image = dict(zip(colmap_files, image_files))
    image_paths = [os.path.join(image_dir, colmap_to_image[f])
                     for f in image_names]
    images = [load_image(x) for x in image_paths]
    images = np.stack(images, axis=0)
    test_indices = np.arange(images.shape[0])[::llffhold]
    test_images = images[test_indices]
    
    return test_images

def load_image(path):
    return cv2.imread(path)[:,:,::-1]

def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
# def im2tensor(image, imtype=np.uint8, cent=1., factor=1.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))
_ = torch.manual_seed(202208)
compute_lpips = LPIPS()

# prediction path
path = '/mnt/sda/experiments/cvpr23/Mip-NeRF-360/logs_Mip-NeRF-360/Scan*/test_preds'
# ground truth path
gt_path_root = '/mnt/sda/experiments/cvpr23_real_cap_dataset'
basedir = '/'.join(path.split('/')[:-2])

results_list = sorted(glob.glob(path))
results_dict = {}
for scene in results_list:
    scene_name = scene.split('/')[-2].split('_')[-1]
    gt_images = load_gts(os.path.join(gt_path_root, scene_name))
    img_list = glob.glob(os.path.join(scene, 'color_0*.png'))
    scene_dict  = {}
    psnrs, ssims, lpipss = [], [], []
    print (f'eval {scene_name}')
    for pred_path in tqdm(img_list):
        img_name = pred_path.split('/')[-1]
        idx = int(img_name.split('.')[0].split('_')[-1])
        
        pred = load_image(pred_path)
        gt   = gt_images[idx]
        
        gt_out = pred_path.replace('color', 'gt')
        cv2.imwrite(gt_out, gt[..., ::-1])
        
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
with open(os.path.join(basedir, 'psnr_ssim_lpips_llff.json'), 'w') as f:
    json.dump(results_dict, f)