import cv2
import numpy as np
import math

def draw_matches(img_A, img_B, keypoints0, keypoints1):
    
    p1s = []
    p2s = []
    dmatches = []
    sum_dist = 0
    for i, (x1, y1) in enumerate(keypoints0):
         
        p1s.append(cv2.KeyPoint(x1, y1, 1))
        p2s.append(cv2.KeyPoint(keypoints1[i][0], keypoints1[i][1], 1))
        dist = math.sqrt((x1 - keypoints1[i][0])**2 + (y1 - keypoints1[i][1])**2)
        sum_dist += dist
        j = len(p1s) - 1
        dmatches.append(cv2.DMatch(j, j, 1))
        
    matched_images = cv2.drawMatches(cv2.cvtColor(img_A, cv2.COLOR_RGB2BGR), p1s, 
                                     cv2.cvtColor(img_B, cv2.COLOR_RGB2BGR), p2s, dmatches, None)
    
    return matched_images, len(dmatches), sum_dist

def align_with_dfm(fm, img_to_align, img_target, image_name, target_name, out_path_debug):
    H, H_init, points_I, points_T, img_aligned, img_T = fm.match(img_target, img_to_align, False)
    keypoints_I = points_I.T
    keypoints_T = points_T.T

    matches_img, nb_matches, sum_dist = draw_matches(img_target, img_to_align, keypoints_I, keypoints_T)
    cv2.imwrite(f"{out_path_debug}/{image_name}_{target_name}_DFM_matches.jpg", matches_img)
    aligned_ov = img_aligned.copy()
    cv2.addWeighted(cv2.cvtColor(img_T, cv2.COLOR_RGB2BGR), 0.5, img_aligned, 0.5, 0, aligned_ov)
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

    indicators = {
        "homography_det": np.linalg.det(H),
        "homography_norm": np.linalg.norm(H),
        "percent_covering": per_white_pix,
        "nb_keypoints": nb_matches,
        "mean_dist_between_keypoints": sum_dist/nb_matches,
        "projected_center_intensity": maxVal,
        "projected_center_location_dist_ratio": new_center_loc_ratio
        }
    
    return img_aligned, indicators