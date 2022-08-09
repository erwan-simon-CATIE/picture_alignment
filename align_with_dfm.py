import cv2
import numpy as np
import math

from utils import draw_matches

def align_with_dfm(fm, img_to_align, img_target, image_name, target_name, out_path_debug):
    H, H_init, points_I, points_T, img_aligned, _ = fm.match(img_target, img_to_align, False)
    keypoints_I = points_I.T
    keypoints_T = points_T.T

    img_to_align = cv2.cvtColor(img_to_align, cv2.COLOR_RGB2BGR)
    img_target = cv2.cvtColor(img_target, cv2.COLOR_RGB2BGR)
    img_aligned = cv2.cvtColor(img_aligned, cv2.COLOR_RGB2BGR)

    matches_img, nb_matches, sum_dist = draw_matches(img_to_align, img_target, keypoints_I, keypoints_T)
    cv2.imwrite(f"{out_path_debug}/{image_name}_{target_name}_DFM_matches.jpg", matches_img)
    aligned_ov = img_aligned.copy()
    cv2.addWeighted(img_target, 0.5, img_aligned, 0.5, 0, aligned_ov)
    cv2.imwrite(f"{out_path_debug}/{image_name}_{target_name}_overlay.jpg", aligned_ov)

    # Indicators calculation
    heightI, widthI, _ = img_to_align.shape
    heightT, widthT, _ = img_target.shape
    white = np.zeros([heightI, widthI, 1],dtype=np.uint8)
    white.fill(255)
    black = np.zeros([heightI, widthI, 3],dtype=np.uint8)
    black.fill(0)
    black[int(heightI/2)-1, int(widthI/2)-1] = [255,255,255]
    black[int(heightI/2)-1, int(widthI/2)] = [255,255,255]
    black[int(heightI/2), int(widthI/2)-1] = [255,255,255]
    black[int(heightI/2), int(widthI/2)] = [255,255,255]
    aligned_mask = cv2.warpPerspective(white, H, (widthT, heightT))
    aligned_points = cv2.warpPerspective(black, H, (widthT, heightT))
    aligned_points = cv2.cvtColor(aligned_points, cv2.COLOR_BGR2GRAY)
    
    height, width = aligned_points.shape

    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(aligned_points)
    new_center_loc_ratio = abs(maxLoc[0]/ width -0.5), abs(maxLoc[1]/ height -0.5)

    n_white_pix = np.sum(aligned_mask == 255)
    per_white_pix = 100 * n_white_pix / (heightT * widthT)
    
    mean_dist = 1000
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
    
    return img_aligned, indicators