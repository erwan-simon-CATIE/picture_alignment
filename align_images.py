# coding: utf-8

import numpy as np
import imutils
import cv2
import os
import math
import traceback

import json
import glob
from shutil import copyfile
import time


def search_array(pixel, img):
    # Faster than iterating over pixels
    pixel_tile = np.tile(pixel, (*img.shape[:2], 1))
    diff = np.sum(np.abs(img - pixel_tile), axis=2)
    last_idx = -1
    for idx in np.argwhere(diff == 0):
        print(f"Found {pixel} at {idx}, value {diff}")
        last_idx = idx
    print (len(np.argwhere(diff == 0)), last_idx)
    return len(np.argwhere(diff == 0)), last_idx


def align_images(image, template, method="SIFT", norm_name="L2", keep_percent=0.3, threshold_dist=1.5):
    
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    if norm_name == "L1":
        norm=cv2.NORM_L1
    if norm_name == "L2":
        norm=cv2.NORM_L2

    if method == "ORB":
        orb = cv2.ORB_create(400)
        (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
        (kpsB, descsB) = orb.detectAndCompute(templateGray, None)
        norm = cv2.NORM_HAMMING
    elif method == "SIFT":
        sift = cv2.xfeatures2d.SIFT_create(2000)
        kpsA, descsA = sift.detectAndCompute(imageGray, None)
        kpsB, descsB = sift.detectAndCompute(templateGray, None)
    elif method == "SURF":
        surf = cv2.xfeatures2d.SURF_create(500, extended=True) # not maxfeatures but hessianThreshold
        kpsA, descsA = surf.detectAndCompute(imageGray, None)
        kpsB, descsB = surf.detectAndCompute(templateGray, None)
    elif method == "BRISK":
        brisk = cv2.BRISK_create()
        kpsA, descsA = brisk.detectAndCompute(imageGray, None)
        kpsB, descsB = brisk.detectAndCompute(templateGray, None)
        norm = cv2.NORM_HAMMING

    bf = cv2.BFMatcher(norm, crossCheck=True)
    matches = bf.match(descsA, descsB)	

    # Sort the matches by their distance (the smaller the distance,
    #   the "more similar" the features are)
    matches = sorted(matches, key=lambda x:x.distance)

    # Keep only the top matches
    keep = int(len(matches) * keep_percent)
    matches = matches[:keep]

    # Allocate memory for the keypoints (x,y-coordinates) from the top matches
    #   -- these coordinates will be used to compute our homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    heightA, widthA, channelsA = image.shape
    heightB, widthB, channelsB = template.shape
    matches_filtered = []
    sum_dist = 0
    # Loop over the top matches
    for (i, m) in enumerate(matches):
        ptA = kpsA[m.queryIdx].pt
        ptB = kpsB[m.trainIdx].pt
        ptAx = ptA[0] / widthA
        ptAy = ptA[1] / heightA
        ptBx = ptB[0] / widthB
        ptBy = ptB[1] / heightB        
        dist = math.sqrt((ptAx - ptBx)**2 + (ptAy - ptBy)**2)
        sum_dist += dist
        # print(i, ptsA[i], ptsB[i], (ptAx, ptAy), (ptBx, ptBy), dist)
        if dist < threshold_dist:
            # Indicate that the two keypoints in the respective images  map to each other
            # if ptAy > 0.7 or ptBy > 0.7: #North Narrabeen fix that does not work
            #     pass
            # else:
            ptsA[i] = ptA
            ptsB[i] = ptB
            matches_filtered.append(m)
    nb_keypoints = len(matches)
    if nb_keypoints > 0:
        mean_dist = sum_dist/nb_keypoints
    else:
        mean_dist = 1

    # print(f"Number of keypoints: {len(matches)}, {len(matches_filtered)}")
    matchedVis = cv2.drawMatches(image, kpsA, template, kpsB, matches_filtered, None)
    matchedVis = imutils.resize(matchedVis, width=1900)

    # Compute the homography matrix between the two sets of matched points
    if len(matches) > 0:
        (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)
    else:
        raise ValueError("No matches")
    try:
        homo_norm = np.linalg.norm(H)
    except TypeError as e:
        print("Exception occured, norm error:", e)
        print(traceback.format_exc())
        raise ValueError("Norm inapplicable")

    print(f"Homography matrix norm: {homo_norm}")

    print(f"Homography det: {str(np.linalg.det(H))}")
    print("--------")
    
    white = np.zeros([heightA, widthA, 1],dtype=np.uint8)
    white.fill(255)
    black = np.zeros([heightA, widthA, 3],dtype=np.uint8)
    black.fill(0)
    black[int(heightA/2)-1, int(widthA/2)-1] = [255,255,255]
    black[int(heightA/2)-1, int(widthA/2)] = [255,255,255]
    black[int(heightA/2), int(widthA/2)-1] = [255,255,255]
    black[int(heightA/2), int(widthA/2)] = [255,255,255]
    aligned = cv2.warpPerspective(image, H, (widthB, heightB))
    aligned_mask = cv2.warpPerspective(white, H, (widthB, heightB))
    aligned_points = cv2.warpPerspective(black, H, (widthB, heightB))
    aligned_points = cv2.cvtColor(aligned_points, cv2.COLOR_BGR2GRAY)
    
    height, width = aligned_points.shape

    (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(aligned_points)
    new_center_loc_ratio = abs(maxLoc[0]/ width -0.5), abs(maxLoc[1]/ height -0.5)

    n_white_pix = np.sum(aligned_mask == 255)
    per_white_pix = 100 * n_white_pix / (heightB * widthB)
    print(f"percent of covering: {per_white_pix}")
    print(f"Keypoints number: {nb_keypoints}")
    print(f"Mean distance: {mean_dist}")
    print(f"New center intensity value: {maxVal}, new center distance ratio {new_center_loc_ratio}")

    return aligned, homo_norm, per_white_pix, nb_keypoints, mean_dist, maxVal, new_center_loc_ratio, matchedVis


def batch_test():
    start_time = time.time()
    default_method_name = "SIFT"
    default_norm_name = "L2"
    threshold_dist = 1.5
    # folder_path = "./images/Lacanau_Kayok_VueNord"
    # folder_path = "./images/ucalib_examples"
    # folder_path = "./images/Lacanau_Kayok_VueNord (copie)"
    # folder_path = "./images/SaintJeanDeLuz_Lafitenia_VueNord"
    # folder_path = "./images/Capbreton_Santocha_VueSud"
    folder_path = "./images/Manly"
    # folder_path = "./images/North_Narrabeen"
    # folder_path = "./images/test_coastsnap"
    # folder_path = "./images/test_rapide2"
    out_path = folder_path.replace("./images", "./results")
    try:
        os.makedirs(out_path)    
        print("Directory " , out_path,  " created ")
    except FileExistsError:
        pass
    
    out_path_debug = out_path + "/debug"
    try:
        os.makedirs(out_path_debug)    
        print("Directory " , out_path_debug,  " created ")
    except FileExistsError:
        pass
    
    # template_path = "./images/Lacanau_Kayok_VueNord/20201204_144234.jpg"
    # template_path = "./images/ucalib_examples/image000007.png"
    # template_path = "./images/SaintJeanDeLuz_Lafitenia_VueNord/20210308_111406.jpg"
    # template_path = "./images/Capbreton_Santocha_VueSud/IMG_20210409_101128.jpg"
    template_path = "./images/Manly/tp9pzlhrd0pfdwyfpptj6czxmiaq8554.jpg"

    template_path_4_3 = None
    template_path_16_9 = None

    # template_path = None
    # template_path_4_3 = "./images/North_Narrabeen/4gf0l6xp79st2bukc739xhzqw5tchopv.jpg"
    # template_path_16_9 = "./images/North_Narrabeen/1m81bw22qltgx1bj1y8pfsy1a8gym72y.jpg"
    
    # template_path = None
    # template_path = "./images/test_rapide/north0.jpg"
    # template_path_4_3 = "./images/test_rapide4/north0_4_3.jpg"
    # template_path_16_9 = "./images/test_rapide4/north0_16_9.jpg"

    if template_path is None:
        template_name_4_3 = os.path.splitext(os.path.basename(template_path_4_3))[0]
        template_name_16_9 = os.path.splitext(os.path.basename(template_path_16_9))[0]

        copyfile(template_path_4_3, f"{out_path}/template_{template_name_4_3}.jpg")
        copyfile(template_path_16_9, f"{out_path}/template_{template_name_16_9}.jpg")
        
        template_4_3 = cv2.imread(template_path_4_3)
        template_16_9 = cv2.imread(template_path_16_9)

        print(f"Template name 4/3: {template_name_4_3}")
        print(f"Template name 16/9: {template_name_16_9}")
    else:
        template_name = os.path.splitext(os.path.basename(template_path))[0]
        copyfile(template_path, f"{out_path}/template_{template_name}.jpg")
        template = cv2.imread(template_path)
        print(f"Template name: {template_name}")

    if os.path.isfile(out_path + '/scores.json'):
        try:
            with open(out_path + '/scores.json', encoding="utf-8") as json_file:
                scores = json.load(json_file)
        except FileNotFoundError as e:
                print("Exception occured, FileNotFoundError:", e)
                print(traceback.format_exc())
                scores = {}
    else:
        scores = {}

    images = sorted(glob.glob(folder_path +'/*.jpg'))
    nb_images = len(images)
    print(f"{nb_images} images found.")
    old_template_path = template_path

    for count, image_path in enumerate(images):
        print("-----------------")
        print(f"Image number {count}/{len(images)}")

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image = cv2.imread(image_path)
        height, width, channels = image.shape
        ratio = width/height 

        if old_template_path is None:
            if abs(ratio - 4/3) <= abs(ratio - 16/9):
                template_path = template_path_4_3
                template_name = template_name_4_3
                template = template_4_3
            else:
                template_path = template_path_16_9
                template_name = template_name_16_9
                template = template_16_9

        print(f"Aligning image {image_name}")
    
        if image_path == template_path:
            print("Template, skipping")
            pass
        else:
            scores[image_name] = {}
            tried = []
            tried_realignment = False
            best_method_name = default_method_name
            best_norm_name = default_norm_name
            
            try:
                best_aligned, best_homo_norm, best_covering, best_nb_keypoints, best_mean_dist, best_new_cent_int, best_new_cent_loc_ratio = align_and_write(image, template, image_name, template_name, best_method_name, best_norm_name, out_path_debug, threshold_dist)
            except ValueError as e:
                print("Exception occured, ValueError:", e)
                print(traceback.format_exc())
                scores[image_name].update({"aligned": False, "is_fake": True, "exception": str(e)})
                with open(out_path + '/scores.json', 'w', encoding="utf-8") as outfile:
                    json.dump(scores, outfile, indent=4)
                continue

            tried.append({
                        "method": best_method_name, 
                        "norm": best_norm_name,
                        "homography_norm_value": best_homo_norm, 
                        "%_covering": best_covering,
                        "nb_keypoints": best_nb_keypoints,
                        "mean_dist": best_mean_dist,
                        "new_cent_int": best_new_cent_int,
                        "new_cent_loc_ratio": best_new_cent_loc_ratio
                    })
            if best_homo_norm >= 150 or best_covering <= 90 or best_mean_dist >= 0.30 or best_new_cent_loc_ratio[0] >= 0.08 or best_new_cent_loc_ratio[1] >= 0.08:
                print(f"Searching for best alignment method")
                for (method_name, norm_name) in [("SURF", "L1"), ("SURF", "L2"), ("SIFT", "L1"), ("SIFT", "L2"), ("BRISK", "HAMMING")]:
                    if method_name == default_method_name and norm_name == default_norm_name:
                        continue
                    try:
                        aligned, homo_norm, covering, nb_keypoints, mean_dist, new_cent_int, new_cent_loc_ratio = align_and_write(image, template, image_name, template_name, method_name, norm_name, out_path_debug, threshold_dist)
                    except ValueError as e:
                        print("Exception occured, ValueError:", e)
                        print(traceback.format_exc())
                        scores[image_name].update({"aligned":False, "is_fake": True, "exception": str(e)})
                        if len(tried) > 0:
                            scores[image_name].update({"tried": tried})
                        with open(out_path + '/scores.json', 'w', encoding="utf-8") as outfile:
                            json.dump(scores, outfile, indent=4)
                        continue
                    tried.append({
                        "method": method_name, 
                        "norm": norm_name,
                        "homography_norm_value": homo_norm, 
                        "%_covering": covering,
                        "nb_keypoints": nb_keypoints,
                        "mean_dist": mean_dist,
                        "new_cent_int": new_cent_int,
                        "new_cent_loc_ratio": new_cent_loc_ratio
                    })
                    if homo_norm < best_homo_norm:
                        best_homo_norm = homo_norm
                        best_aligned = aligned
                        best_method_name = method_name
                        best_norm_name = norm_name
                        best_covering = covering
                        best_nb_keypoints = nb_keypoints
                        best_mean_dist = mean_dist
                        best_new_cent_int = new_cent_int
                        best_new_cent_loc_ratio = new_cent_loc_ratio
            if best_homo_norm > 500 or best_mean_dist > 0.4:
                # Try to realigned the aligned image with the image
                try:
                    realigned, homo_norm_re, covering_re, nb_keypoints_re, mean_dist_re, new_cent_int_re, new_cent_loc_ratio_re = align_and_write(aligned, image, image_name, template_name, best_method_name, best_norm_name, out_path_debug, threshold_dist)
                except ValueError as e:
                        print("Exception occured, ValueError:", e)
                        print(traceback.format_exc())
                        realigned = None
                        homo_norm_re = 9999
                        covering_re = 0
                        nb_keypoints_re = 0
                        mean_dist_re = 100
                        new_cent_int_re = 0
                        new_cent_loc_ratio_re = (0.5,0.5)

                tried_realignment = True
                print(f"Realigned with homo_norm_re {homo_norm_re}, covering_re {covering_re}, nb_keypoints_re {nb_keypoints_re}, mean_dist_re {mean_dist_re}")
                cv2.imwrite(f"{out_path_debug}/{image_name}_{template_name}_realigned.jpg", realigned)
                # if homo_norm_re > 700 or mean_dist_re >= 0.3:
                #     print(f"Image {image_name} is not a photo of the template beach...")
                #     scores[image_name] = {
                #         "template": template_name,
                #         "aligned": False,
                #         "is_fake": True,
                #         "method": best_method_name, 
                #         "norm": best_norm_name,
                #         "homography_norm_value": best_homo_norm, 
                #         "%_covering": best_covering,
                #         "nb_keypoints": best_nb_keypoints,
                #         "mean_dist": best_mean_dist,
                #         "new_cent_int": best_new_cent_int,
                #         "new_cent_loc_ratio": best_new_cent_loc_ratio,
                #         "realignment": {
                #             "homography_norm_value_re": homo_norm_re, 
                #             "percent_covering_re": covering_re,
                #             "nb_keypoints_re": nb_keypoints_re,
                #             "mean_dist_re": mean_dist_re,
                #             "new_cent_int_re": new_cent_int_re,
                #             "new_cent_loc_ratio_re": new_cent_loc_ratio_re
                #         }
                #     }
                #     if len(tried) > 0:
                #         scores[image_name].update({"tried": tried})
                #     with open(out_path + '/scores.json', 'w', encoding="utf-8") as outfile:
                #         json.dump(scores, outfile, indent=4)
                #     continue

            if (best_homo_norm > 1500 or best_mean_dist >= 0.5 or best_new_cent_int <= 100 or (best_new_cent_loc_ratio[0] > 0.11 and best_new_cent_loc_ratio[1] > 0.11)):
                print(f"Image {image_name} is not a photo of the template beach...")
                scores[image_name] = {
                    "template": template_name,
                    "aligned": False,
                    "is_fake": True,
                    "method": best_method_name, 
                    "norm": best_norm_name,
                    "homography_norm_value": best_homo_norm, 
                    "%_covering": best_covering,
                    "nb_keypoints": best_nb_keypoints,
                    "mean_dist": best_mean_dist,
                    "new_cent_int": best_new_cent_int,
                    "new_cent_loc_ratio": best_new_cent_loc_ratio,
                }
            elif (best_homo_norm > 500 or best_covering <= 80 or best_mean_dist >= 0.4 or best_new_cent_loc_ratio[0] > 0.05 or best_new_cent_loc_ratio[1] > 0.05):
                print(f"Failed to align image {image_name}, but nevertheless, it seems to be the right beach image")
                cv2.imwrite(f"{out_path}/{image_name}_{template_name}_tried_to_be_aligned_with_{best_method_name}_{best_norm_name}_.jpg", best_aligned)
                scores[image_name] = {
                    "template":template_name,
                    "aligned": False,
                    "is_fake": False,
                    "method": best_method_name, 
                    "norm": best_norm_name, 
                    "homography_norm_value": best_homo_norm, 
                    "%_covering": best_covering,
                    "nb_keypoints": best_nb_keypoints,
                    "mean_dist": best_mean_dist, 
                    "new_cent_int": best_new_cent_int,
                    "new_cent_loc_ratio": best_new_cent_loc_ratio
                }
            else:
                print(f"Image {image_name} aligned with {best_method_name}, {best_norm_name} with score {best_homo_norm}")
                cv2.imwrite(f"{out_path}/{image_name}_{template_name}_aligned_with_{best_method_name}_{best_norm_name}_.jpg", best_aligned)
                scores[image_name] = {
                    "template": template_name,
                    "aligned": True,
                    "is_fake": False,
                    "method": best_method_name, 
                    "norm": best_norm_name,
                    "homography_norm_value": best_homo_norm, 
                    "%_covering": best_covering,
                    "nb_keypoints": best_nb_keypoints,
                    "mean_dist": best_mean_dist,
                    "new_cent_int": best_new_cent_int,
                    "new_cent_loc_ratio": best_new_cent_loc_ratio
                }
            if tried_realignment:
                scores[image_name].update({
                    "realignment": {
                        "homography_norm_value_re": homo_norm_re, 
                        "percent_covering_re": covering_re,
                        "nb_keypoints_re": nb_keypoints_re,
                        "mean_dist_re": mean_dist_re,
                        "new_cent_int_re": new_cent_int_re,
                        "new_cent_loc_ratio_re": new_cent_loc_ratio_re
                    }})
            if len(tried) > 0:
                        scores[image_name].update({"tried": tried})
            aligned_ov = best_aligned.copy()
            cv2.addWeighted(template, 0.5, best_aligned, 0.5, 0, aligned_ov)
            cv2.imwrite(f"{out_path}/{image_name}_{template_name}_{best_method_name}_{best_norm_name}_overlay.jpg", aligned_ov)
            with open(out_path + '/scores.json', 'w', encoding="utf-8") as outfile:
                json.dump(scores, outfile, indent=4)
    if os.path.isfile(out_path + '/scores.json'):
        try:
            with open(out_path + '/scores.json', encoding="utf-8") as json_file:
                scores = json.load(json_file)
                values_homo = []
                values_cov = []
                values_keypoints = []
                values_mean_dist = []
                values_new_cent_int = []
                values_new_cent_loc_ratio = []
                for item in scores.items():
                    try:
                        print(item)
                        values_homo.append(item[1]["homography_norm_value"])
                        values_cov.append(item[1]["%_covering"])
                        values_keypoints.append(item[1]["nb_keypoints"])
                        values_mean_dist.append(item[1]["mean_dist"])
                        values_new_cent_int.append(item[1]["new_cent_int"])
                        values_new_cent_loc_ratio.append(item[1]["new_cent_loc_ratio"])
                    except KeyError as e:
                        print("Exception occured, Key error:", e)
                        print(traceback.format_exc())
                        continue
                scores["stats"] = {}
                if len(values_homo) > 0:
                    scores["stats"]["homography_norm_value"] = {
                            "min": str(np.min(values_homo)),
                            "max": str(np.max(values_homo)),
                            "mean": str(np.mean(values_homo)),
                            "median": str(np.median(values_homo))
                        }
                if len(values_cov) > 0:
                    scores["stats"]["%_covering"] = {
                            "min": str(np.min(values_cov)),
                            "max": str(np.max(values_cov)),
                            "mean": str(np.mean(values_cov)),
                            "median": str(np.median(values_cov))
                        }
                if len(values_keypoints) > 0:
                    scores["stats"]["nb_keypoints"] = {
                            "min": str(np.min(values_keypoints)),
                            "max": str(np.max(values_keypoints)),
                            "mean": str(np.mean(values_keypoints)),
                            "median": str(np.median(values_keypoints))
                        }
                if len(values_mean_dist) > 0:
                    scores["stats"]["mean_dist"] = {
                            "min": str(np.min(values_mean_dist)),
                            "max": str(np.max(values_mean_dist)),
                            "mean": str(np.mean(values_mean_dist)),
                            "median": str(np.median(values_mean_dist))
                        }
                if len(values_new_cent_int) > 0:
                    scores["stats"]["mean_dist"] = {
                            "min": str(np.min(values_new_cent_int)),
                            "max": str(np.max(values_new_cent_int)),
                            "mean": str(np.mean(values_new_cent_int)),
                            "median": str(np.median(values_new_cent_int))
                        }
                if len(values_new_cent_loc_ratio) > 0:
                    scores["stats"]["mean_dist"] = {
                            "min": str(np.min(values_new_cent_loc_ratio)),
                            "max": str(np.max(values_new_cent_loc_ratio)),
                            "mean": str(np.mean(values_new_cent_loc_ratio)),
                            "median": str(np.median(values_new_cent_loc_ratio))
                        }
                scores["stats"].update({
                    "comp_time_minutes": (time.time() - start_time) / 60
                })
                with open(out_path + '/scores.json', 'w', encoding="utf-8") as outfile:
                    json.dump(scores, outfile, indent=4)
        except FileNotFoundError as e:
            print("Exception occured, FileNotFoundError:", e)
            print(traceback.format_exc())


def align_and_write(image, template, image_name, template_name, method_name, norm_name, out_path_debug, threshold_dist):
    debug_image_name = f"{image_name}_{template_name}_{method_name}_{norm_name}_descriptors_matching.jpg"

    aligned, homo_norm, covering, nb_keypoints, mean_dist, new_cent_int, new_cent_loc_ratio, matchedVis = align_images(image, template, method=method_name, norm_name=norm_name, threshold_dist=threshold_dist)
    cv2.imwrite(f"{out_path_debug}/{debug_image_name}", matchedVis)

    aligned_ov = aligned.copy()
    cv2.addWeighted(template, 0.5, aligned, 0.5, 0, aligned_ov)
    cv2.imwrite(f"{out_path_debug}/{image_name}_{template_name}_{method_name}_{norm_name}_overlay.jpg", aligned_ov)
    
    return aligned, homo_norm, covering, nb_keypoints, mean_dist, new_cent_int, new_cent_loc_ratio


if __name__ == "__main__":
    batch_test()
