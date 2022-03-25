# coding: utf-8

import cv2
import numpy as np
import math
import imutils
import traceback

def align_images(image, image_gray, target, target_gray, target_mask, method, norm_name, method_param, 
        keep_percent, threshold_dist, use_mask, findPlanarTransformation):
    print("Aligning with", method, norm_name, method_param)
    heightT, widthT, _ = target.shape
    heightI, widthI, _ = image.shape

    if norm_name == "L1":
        norm=cv2.NORM_L1
    if norm_name == "L2":
        norm=cv2.NORM_L2

        
    mask = np.array(target_mask != 255, dtype=np.uint8)

    if method == "ORB":
        orb = cv2.ORB_create(method_param)
        (kpsI, descsI) = orb.detectAndCompute(image_gray, None)
        (kpsT, descsT) = orb.detectAndCompute(target_gray, None)
        norm = cv2.NORM_HAMMING
    elif method == "SIFT":
        sift = cv2.xfeatures2d.SIFT_create(method_param)
        kpsI, descsI = sift.detectAndCompute(image_gray, None)
        kpsT, descsT = sift.detectAndCompute(target_gray, None)
    elif method == "SURF":
        surf = cv2.xfeatures2d.SURF_create(method_param, extended=True) # not maxfeatures but hessianThreshold
        kpsI, descsI = surf.detectAndCompute(image_gray, None)
        kpsT, descsT = surf.detectAndCompute(target_gray, None)
    elif method == "BRISK":
        brisk = cv2.BRISK_create(method_param)
        kpsI, descsI = brisk.detectAndCompute(image_gray, None)
        kpsT, descsT = brisk.detectAndCompute(target_gray, None)
        norm = cv2.NORM_HAMMING
    else:
        raise ValueError("Unknown method name: " + str(method))

    bf = cv2.BFMatcher(norm, crossCheck=True)
    try:
        matches = bf.match(descsI, descsT)	
    except cv2.error as e:
        print("OpenCV Error:", e)
        raise ValueError("OpenCV Error: " + str(e))

    # Sort the matches by their distance (the smaller the distance,
    #   the "more similar" the features are)
    matches = sorted(matches, key=lambda x:x.distance)

    # Keep only the top matches
    keep = int(len(matches) * keep_percent)
    matches = matches[:keep]

    # Allocate memory for the keypoints (x,y-coordinates) from the top matches
    #   -- these coordinates will be used to compute our homography matrix
    ptsI = np.zeros((len(matches), 2), dtype="float")
    ptsT = np.zeros((len(matches), 2), dtype="float")

    matches_filtered = []
    sum_dist = 0
    # Loop over the top matches
    for (i, m) in enumerate(matches):
        ptI = kpsI[m.queryIdx].pt
        ptT = kpsT[m.trainIdx].pt
        ptIx = ptI[0] / widthI
        ptIy = ptI[1] / heightI
        ptTx = ptT[0] / widthT
        ptTy = ptT[1] / heightT  
        dist = math.sqrt((ptIx - ptTx)**2 + (ptIy - ptTy)**2)
        sum_dist += dist
        # print(i, ptsI[i], ptsT[i], (ptIx, ptIy), (ptTx, ptTy), dist)
        if dist < threshold_dist:
            # Indicate that the two keypoints in the respective images  map to each other
            # if ptIy > 0.7 or ptTy > 0.7: #North Narrabeen fix that does not work
            #     pass
            # else:
            if (use_mask and target_mask is not None):
                ptTrx = round(ptT[0])
                # TODO Caution if ptTrx > widthT
                if ptTrx == widthT:
                    ptTrx -= 1
                elif ptTrx == -1:
                    ptTrx = 0

                ptTry = round(ptT[1])
                # TODO Caution if ptTry > heightT
                if ptTry == heightT:
                    ptTry -= 1
                elif ptTry == -1:
                    ptTry = 0
                
                # ptIrx = round(ptI[0])
                # # TODO Caution if ptIrx > widthT
                # if ptIrx == widthI:
                #     ptIrx -= 1
                # elif ptIrx == -1:
                #     ptIrx = 0

                # ptIry = round(ptI[1])
                # # TODO Caution if ptIry > heightT
                # if ptIry == heightI:
                #     ptIry -= 1
                # elif ptIry == -1:
                #     ptIry = 0

                if target_mask[ptTry, ptTrx] == 0:# and target_mask[ptIry, ptIrx] == 0:
                    ptsI[i] = ptI
                    ptsT[i] = ptT
                    matches_filtered.append(m)
            else:
                ptsI[i] = ptI
                ptsT[i] = ptT
                matches_filtered.append(m)

    nb_keypoints = len(matches_filtered)
    print(f"Number of keypoints: {nb_keypoints}")
    if nb_keypoints > 0:
        mean_dist = sum_dist/nb_keypoints
    else:
        mean_dist = 1

    # print(f"Number of keypoints: {len(matches)}, {len(matches_filtered)}")
    matchedVis = cv2.drawMatches(image, kpsI, target, kpsT, matches_filtered, None)
    matchedVis = imutils.resize(matchedVis, width=1900)

    # Compute the homography matrix between the two sets of matched points
    if len(matches_filtered) > 0:
        if findPlanarTransformation:
            print("findHomography")
            (H, mask) = cv2.findHomography(ptsI, ptsT, method=cv2.RANSAC)
        else:
            print("findFundamentalMat")
            (H, mask) = cv2.findFundamentalMat(ptsI, ptsT, method=cv2.FM_RANSAC)
        print(H)
    else:
        raise ValueError("No matches")
    if H is None:
        raise ValueError("No homography found")
    try:
        homo_norm = np.linalg.norm(H)
    except TypeError as e:
        print("Exception occured, norm error:", e)
        print(traceback.format_exc())
        raise ValueError("Norm inapplicable")

    print(f"Homography det: {str(np.linalg.det(H))}")
    print(f"Homography matrix norm: {homo_norm}")
    
    white = np.zeros([heightI, widthI, 1],dtype=np.uint8)
    white.fill(255)
    black = np.zeros([heightI, widthI, 3],dtype=np.uint8)
    black.fill(0)
    black[int(heightI/2)-1, int(widthI/2)-1] = [255,255,255]
    black[int(heightI/2)-1, int(widthI/2)] = [255,255,255]
    black[int(heightI/2), int(widthI/2)-1] = [255,255,255]
    black[int(heightI/2), int(widthI/2)] = [255,255,255]
    aligned = cv2.warpPerspective(image, H, (widthT, heightT))
    aligned_mask = cv2.warpPerspective(white, H, (widthT, heightT))
    aligned_points = cv2.warpPerspective(black, H, (widthT, heightT))
    aligned_points = cv2.cvtColor(aligned_points, cv2.COLOR_BGR2GRAY)
    
    height, width = aligned_points.shape

    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(aligned_points)
    new_center_loc_ratio = abs(maxLoc[0]/ width -0.5), abs(maxLoc[1]/ height -0.5)

    n_white_pix = np.sum(aligned_mask == 255)
    per_white_pix = 100 * n_white_pix / (heightT * widthT)
    print(f"percent of covering: {per_white_pix}")
    print(f"Mean distance: {mean_dist}")
    print(f"New center intensity value: {maxVal}, new center distance ratio {new_center_loc_ratio}")

    indicators = {
        "homography_det": np.linalg.det(H),
        "homography_norm": homo_norm,
        "percent_covering": per_white_pix,
        "nb_keypoints": nb_keypoints,
        "mean_dist_between_keypoints": mean_dist,
        "projected_center_intensity": maxVal,
        "projected_center_location_dist_ratio": new_center_loc_ratio
        }
    return aligned, matchedVis, indicators


def align_images_v_costsnap(image, image_gray, target, target_gray, target_mask, method, norm_name, method_param, keep_percent, threshold_dist, use_mask):
    print("Aligning with", method, norm_name, method_param)
    heightT, widthT, _ = target.shape
    heightI, widthI, _ = image.shape
        
    mask = np.array(target_mask != 255, dtype=np.uint8)

    orb = cv2.ORB_create(method_param)
    (kpsI, descsI) = orb.detectAndCompute(image_gray, mask=mask)
    (kpsT, descsT) = orb.detectAndCompute(target_gray, mask=mask)
    norm = cv2.NORM_HAMMING

    bf = cv2.BFMatcher(norm, crossCheck=True)
    try:
        matches = bf.match(descsI, descsT)	
    except cv2.error as e:
        print("OpenCV Error:", e)
        raise ValueError("OpenCV Error: " + str(e))

    # Sort the matches by their distance (the smaller the distance,
    #   the "more similar" the features are)
    matches = sorted(matches, key=lambda x:x.distance)

    # Keep only the top matches
    keep = int(len(matches) * keep_percent)
    matches = matches[:keep]

    # Allocate memory for the keypoints (x,y-coordinates) from the top matches
    #   -- these coordinates will be used to compute our homography matrix
    ptsI = np.zeros((len(matches), 2), dtype="float")
    ptsT = np.zeros((len(matches), 2), dtype="float")

    matches_filtered = []
    sum_dist = 0
    # Loop over the top matches
    for (i, m) in enumerate(matches):
        ptI = kpsI[m.queryIdx].pt
        ptT = kpsT[m.trainIdx].pt
        ptIx = ptI[0] / widthI
        ptIy = ptI[1] / heightI
        ptTx = ptT[0] / widthT
        ptTy = ptT[1] / heightT  
        dist = math.sqrt((ptIx - ptTx)**2 + (ptIy - ptTy)**2)
        sum_dist += dist
        # print(i, ptsI[i], ptsT[i], (ptIx, ptIy), (ptTx, ptTy), dist)
        if dist < threshold_dist:
            # Indicate that the two keypoints in the respective images  map to each other
            # if ptIy > 0.7 or ptTy > 0.7: #North Narrabeen fix that does not work
            #     pass
            # else:
            if (use_mask and target_mask is not None):
                ptTrx = round(ptT[0])
                # TODO Caution if ptTrx > widthT
                if ptTrx == widthT:
                    ptTrx -= 1
                elif ptTrx == -1:
                    ptTrx = 0

                ptTry = round(ptT[1])
                # TODO Caution if ptTry > heightT
                if ptTry == heightT:
                    ptTry -= 1
                elif ptTry == -1:
                    ptTry = 0
                

                if target_mask[ptTry, ptTrx] == 0:# and target_mask[ptIry, ptIrx] == 0:
                    ptsI[i] = ptI
                    ptsT[i] = ptT
                    matches_filtered.append(m)
            else:
                ptsI[i] = ptI
                ptsT[i] = ptT
                matches_filtered.append(m)

    nb_keypoints = len(matches_filtered)
    print(f"Number of keypoints: {nb_keypoints}")
    if nb_keypoints > 0:
        mean_dist = sum_dist/nb_keypoints
    else:
        mean_dist = 1

    # print(f"Number of keypoints: {len(matches)}, {len(matches_filtered)}")
    matchedVis = cv2.drawMatches(image, kpsI, target, kpsT, matches_filtered, None)
    matchedVis = imutils.resize(matchedVis, width=1900)

    # Compute the homography matrix between the two sets of matched points
    if len(matches_filtered) > 0:
        (H, mask) = cv2.findHomography(ptsI, ptsT, method=cv2.RANSAC)
    else:
        raise ValueError("No matches")
    if H is None:
        raise ValueError("No homography found")
    try:
        homo_norm = np.linalg.norm(H)
    except TypeError as e:
        print("Exception occured, norm error:", e)
        print(traceback.format_exc())
        raise ValueError("Norm inapplicable")

    print(f"Homography det: {str(np.linalg.det(H))}")
    print(f"Homography matrix norm: {homo_norm}")
   
    aligned = cv2.warpPerspective(image, H, (widthT, heightT))


    indicators = {
        "homography_det": np.linalg.det(H),
        "homography_norm": homo_norm,
        "percent_covering": -1,
        "nb_keypoints": nb_keypoints,
        "mean_dist_between_keypoints": mean_dist,
        "projected_center_intensity": -1,
        "projected_center_location_dist_ratio": -1
    }
    return aligned, matchedVis, indicators

def align_and_write(image, image_gray, target, target_gray, target_mask, image_name, target_name, method_name, norm_name,
     method_param, out_path_debug, keep_percent, threshold_dist, use_mask):
    debug_image_name = f"{image_name}_{target_name}_{method_name}_{norm_name}_descriptors_matching.jpg"

    aligned, matchedVis, indicators = align_images(image, image_gray, target, target_gray, 
        target_mask, method_name, norm_name, method_param, keep_percent, threshold_dist, use_mask)
    cv2.imwrite(f"{out_path_debug}/{debug_image_name}", matchedVis)

    aligned_ov = aligned.copy()

    cv2.addWeighted(target, 0.5, aligned, 0.5, 0, aligned_ov)
    cv2.imwrite(f"{out_path_debug}/{image_name}_{target_name}_{method_name}_{norm_name}_overlay.jpg", 
        aligned_ov)
    
    return aligned, indicators


def align_and_write(image, image_gray, target, target_gray, target_mask, image_name, target_name, method_name, norm_name,
     method_param, out_path_debug, keep_percent, threshold_dist, use_mask, findPlanarTransformation):
    debug_image_name = f"{image_name}_{target_name}_{method_name}_{norm_name}_matches.jpg"

    aligned, matchedVis, indicators = align_images(image, image_gray, target, target_gray, 
        target_mask, method_name, norm_name, method_param, keep_percent, threshold_dist, 
        use_mask, findPlanarTransformation)
    cv2.imwrite(f"{out_path_debug}/{debug_image_name}", matchedVis)

    aligned_ov = aligned.copy()
    cv2.addWeighted(target, 0.5, aligned, 0.5, 0, aligned_ov)
    cv2.imwrite(f"{out_path_debug}/{image_name}_{target_name}_{method_name}_{norm_name}_overlay.jpg", 
        aligned_ov)

    return aligned, indicators