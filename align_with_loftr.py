from configparser import Interpolation
import os
from copy import deepcopy

import torch
import cv2
import numpy as np
import matplotlib.cm as cm
from LoFTR.src.loftr import LoFTR, default_cfg


import cv2
import numpy as np
import math

from utils import image_resize_keep_ratio, draw_matches

 
def align_with_loftr(matcher, img_to_align, img_target, image_name, target_name, out_path_debug, resize_factor = 0.5, max_nb_pixels = 3600000):
    # For square images, max_nb_pixels = 3600000 corresponds to 1900 x 1900 pixels

    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')


    heightI, widthI, _ = img_to_align.shape
    heightT, widthT, _ = img_target.shape

    print("Input size", img_to_align.shape)
    print("Target size", img_target.shape)

    if int(heightI * resize_factor) * int (widthI * resize_factor) > max_nb_pixels:
        # Decreasing resize factor to avoid Runtime Error out of memory 
        resize_factor = 1/ (math.sqrt(heightI * widthI / max_nb_pixels))
        resize_factor = math.floor(resize_factor * 100) / 100

    print("resize_factor", resize_factor)

    if (widthT > 2050) or (heightT > 2050) or (widthI > 2050) or (heightI > 2050):
        img_to_align_min = cv2.resize(img_to_align, (int(widthI * resize_factor), int(heightI * resize_factor)), interpolation=cv2.INTER_CUBIC) 
        img_target_min = cv2.resize(img_target, (int(widthT * resize_factor), int(heightT * resize_factor)), interpolation=cv2.INTER_CUBIC) 
    else:
        resize_factor = 1
        img_to_align_min = img_to_align
        img_target_min = img_target

    heightI_min, widthI_min, _ = img_to_align_min.shape
    heightT_min, widthT_min, _ = img_target_min.shape

    print("Input min size", img_to_align_min.shape)
    print("Target min size", img_target_min.shape)

    print("resize_factor", resize_factor)

    # input size should be divisible by 8
    if heightI_min % 8 != 0:
        print("Croping height image to align")
        img_to_align_min = img_to_align_min[heightI_min % 8:heightI_min, 0:widthI_min]
        print("New shape", img_to_align_min.shape)
    if widthI_min % 8 != 0:
        print("Croping width image to align")
        img_to_align_min = img_to_align_min[0:heightI_min, widthI_min % 8:widthI_min]
        print("New shape", img_to_align_min.shape)

    if heightT_min % 8 != 0:
        print("Warning: Croping height image target")
        img_target_min = img_target_min[heightT_min % 8:heightT_min, 0:widthT_min]
        print("New shape", img_target_min.shape)
    if widthT_min % 8 != 0:
        print("Warning: Croping width image target")
        img_target_min = img_target_min[0:heightT_min, widthT_min % 8:widthT_min]
        print("New shape", img_target_min.shape)

    img_to_align_gray = cv2.cvtColor(img_to_align_min, cv2.COLOR_BGR2GRAY)
    img_target_gray = cv2.cvtColor(img_target_min, cv2.COLOR_BGR2GRAY)

    print("Input min size final", img_to_align_min.shape)
    print("Target min size final", img_target_min.shape)
    
    img_to_align_torch = torch.from_numpy(img_to_align_gray)[None][None].to(device) / 255.
    img_target_torch = torch.from_numpy(img_target_gray)[None][None].to(device) / 255.
    batch = {'image0': img_to_align_torch, 'image1': img_target_torch}

    # Inference with LoFTR and get prediction
    with torch.no_grad():
        matcher(batch)
        mkpts0 = batch['mkpts0_f'].cpu().numpy()
        mkpts1 = batch['mkpts1_f'].cpu().numpy()

    del img_to_align_torch, img_target_torch, batch
    torch.cuda.empty_cache()

    H, _ = cv2.findHomography(mkpts0, mkpts1, cv2.RANSAC)
    H_res = np.array([[H[0][0], H[0][1], (H[0][2]*(1/resize_factor))], [H[1][0], H[1][1], (H[1][2]*(1/resize_factor))], [(H[2][0]*resize_factor), H[2][1]*resize_factor, H[2][2]]])
    
    img_aligned = cv2.warpPerspective(img_to_align, H_res, (widthT, heightT))

    matches_img, nb_matches, sum_dist = draw_matches(img_to_align_min, img_target_min, mkpts0, mkpts1)

    cv2.imwrite(f"{out_path_debug}/{image_name}_{target_name}_LoFTR_matches.jpg", matches_img)
    aligned_ov = img_aligned.copy()
    cv2.addWeighted(img_target, 0.5, img_aligned, 0.5, 0, aligned_ov)
    cv2.imwrite(f"{out_path_debug}/{image_name}_{target_name}_overlay.jpg", aligned_ov)

    # Indicators calculation
    white = np.zeros([heightI, widthI, 1],dtype=np.uint8)
    white.fill(255)
    black = np.zeros([heightI, widthI, 3],dtype=np.uint8)
    black.fill(0)
    black[int(heightI/2)-1, int(widthI/2)-1] = [255,255,255]
    black[int(heightI/2)-1, int(widthI/2)] = [255,255,255]
    black[int(heightI/2), int(widthI/2)-1] = [255,255,255]
    black[int(heightI/2), int(widthI/2)] = [255,255,255]
    aligned_mask = cv2.warpPerspective(white, H_res, (widthT, heightT))
    aligned_points = cv2.warpPerspective(black, H_res, (widthT, heightT))
    aligned_points = cv2.cvtColor(aligned_points, cv2.COLOR_BGR2GRAY)
    
    height, width = aligned_points.shape

    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(aligned_points)
    new_center_loc_ratio = abs(maxLoc[0]/ width -0.5), abs(maxLoc[1]/ height -0.5)

    n_white_pix = np.sum(aligned_mask == 255)
    per_white_pix = 100 * n_white_pix / (heightT * widthT)

    if nb_matches > 0:
        mean_dist = sum_dist/nb_matches
    indicators = {
        "homography_det": np.linalg.det(H),
        "homography_norm": np.linalg.norm(H),
        "percent_covering": per_white_pix,
        "nb_keypoints": nb_matches,
        "mean_dist_between_keypoints": mean_dist,
        "projected_center_intensity": maxVal,
        "projected_center_location_dist_ratio": new_center_loc_ratio
        }

    # cv2.imwrite(image_name + '_aligned.jpg', img_aligned)
    # print("wrote", image_name + '_aligned.jpg')

    return img_aligned, indicators





