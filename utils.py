# coding: utf-8
import os
import json
import traceback
import numpy as np
import time
import cv2
import math

def image_resize_keep_ratio(image, width=None, height=None):
    if width is None and height is None:
        return image

    (h, w) = image.shape[:2]
    
    if height is None:
        r = width / float(w)
        dim = (width, int(h * r))
    else:
        r = height / float(h)
        dim = (int(w * r), height)
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_CUBIC)

    return resized

def draw_matches(img_A, img_B, keypoints0, keypoints1):

    p1s = []
    p2s = []
    dmatches = []
    sum_dist = 0
    hA, wA, _ = img_A.shape
    hB, wB, _ = img_B.shape
    for i, (x1, y1) in enumerate(keypoints0):
            
        p1s.append(cv2.KeyPoint(x1, y1, 1))
        p2s.append(cv2.KeyPoint(keypoints1[i][0], keypoints1[i][1], 1))
        dist = math.sqrt(((x1/wA) - (keypoints1[i][0]/wB))**2 + ((y1/hA) - (keypoints1[i][1]/hB))**2)
        sum_dist += dist
        j = len(p1s) - 1
        dmatches.append(cv2.DMatch(j, j, 1))
    
    matched_images = cv2.drawMatches(img_A, p1s, img_B,p2s, dmatches, None)

    return matched_images, len(dmatches), sum_dist

def export_stats(scores_path, start_time):
   if os.path.isfile(scores_path):
        try:
            with open(scores_path, encoding="utf-8") as json_file:
                scores = json.load(json_file)
                values_det = []
                values_homo = []
                values_cov = []
                values_keypoints = []
                values_mean_dist = []
                values_new_cent_int = []
                values_new_cent_loc_ratio = []
                nb_aligned = 0
                nb_non_aligned = 0
                nb_fake = 0
                for item in scores.items():
                    try:
                        print(item)
                        if item[0] == "stats":
                            continue
                        if "aligned" in item[1]:
                            if item[1]["aligned"]:
                                nb_aligned += 1
                            elif "is_fake" in item[1]:
                                if item[1]["is_fake"]:
                                    nb_fake += 1
                                else:
                                    nb_non_aligned += 1
                        
                        values_det.append(item[1]["homography_det"])
                        values_homo.append(item[1]["homography_norm_value"])
                        values_cov.append(item[1]["%_covering"])
                        values_keypoints.append(item[1]["nb_keypoints"])
                        values_mean_dist.append(item[1]["mean_dist"])
                        values_new_cent_int.append(item[1]["new_cent_int"])
                        values_new_cent_loc_ratio.append(item[1]["new_cent_loc_ratio"])
                    except KeyError as e:
                        # print("Exception occured, Key error:", e)
                        # print(traceback.format_exc())
                        continue
                scores["stats"] = {}
                scores["stats"]["nb_aligned"] = str(nb_aligned)
                scores["stats"]["nb_non_aligned"] = str(nb_non_aligned)
                scores["stats"]["nb_fake"] = str(nb_fake)
                scores["stats"]["total_images"] = str(len(values_homo))
                if len(values_det) > 0:
                    scores["stats"]["homography_det"] = {
                            "min": str(np.min(values_det)),
                            "max": str(np.max(values_det)),
                            "mean": str(np.mean(values_det)),
                            "median": str(np.median(values_det))
                        }
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
                    scores["stats"]["new_cent_int"] = {
                            "min": str(np.min(values_new_cent_int)),
                            "max": str(np.max(values_new_cent_int)),
                            "mean": str(np.mean(values_new_cent_int)),
                            "median": str(np.median(values_new_cent_int))
                        }
                if len(values_new_cent_loc_ratio) > 0:
                    scores["stats"]["new_cent_loc_ratio"] = {
                            "min": str(np.min(values_new_cent_loc_ratio)),
                            "max": str(np.max(values_new_cent_loc_ratio)),
                            "mean": str(np.mean(values_new_cent_loc_ratio)),
                            "median": str(np.median(values_new_cent_loc_ratio))
                        }
                scores["stats"].update({
                    "comp_time_minutes": (time.time() - start_time) / 60
                })
                with open(scores_path, 'w', encoding="utf-8") as outfile:
                    json.dump(scores, outfile, indent=4)
        except FileNotFoundError as e:
            print("Exception occured, FileNotFoundError:", e)
            print(traceback.format_exc())

