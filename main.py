# coding: utf-8

import imutils
import cv2
import os
import traceback
import json
import glob
from shutil import copyfile
import time
import configparser

from align_images import align_and_write
from utils import export_stats


def get_method_param_value(method_name, params):
    if method_name == "ORB":
        method_param = int(params["Alignment"]["orb_max_features"])
    elif method_name == "SIFT":
        method_param = int(params["Alignment"]["sift_max_features"])
    elif method_name == "SURF":
        method_param = int(params["Alignment"]["surf_hessian_threshold"])
    elif method_name == "BRISK":
        method_param = int(params["Alignment"]["brisk_thresh"])
    else:
        method_param = None
    return method_param

def format_parameters_to_try(params):
    values = params["Retry"]["parameters_to_try"].split(",")
    return [value.split("-") for value in values]


def batch_test():
    params = configparser.ConfigParser(allow_no_value=True)
    params.read("parameters.ini")
    print(params["Alignment"]["target_path"])   
    start_time = time.time()
    keep_percent = float(params["Alignment"]["keep_percent"])
    default_method_name = params["Alignment"]["default_method"]
    default_norm_name = params["Alignment"]["default_norm"]
    threshold_dist = float(params["Alignment"]["threshold_dist"])
    folder_path = params["Alignment"]["folder_path"]
    use_mask = params.getboolean("Alignment", "use_mask")
    
    out_path = folder_path + "/results"
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

    target_path = params["Alignment"]["target_path"]
    target_path_4_3 = params["Alignment"]["target_path_4_3"]
    target_path_16_9 = params["Alignment"]["target_path_16_9"]

    target_mask_path = params["Alignment"]["mask_path"]
    target_mask_path_4_3 = params["Alignment"]["mask_path_4_3"]
    target_mask_path_16_9 = params["Alignment"]["mask_path_16_9"]

    if target_path is None:
        target_name_4_3 = os.path.splitext(os.path.basename(target_path_4_3))[0]
        target_name_16_9 = os.path.splitext(os.path.basename(target_path_16_9))[0]

        copyfile(target_path_4_3, f"{out_path}/target_{target_name_4_3}.jpg")
        copyfile(target_path_16_9, f"{out_path}/target_{target_name_16_9}.jpg")
        
        target_4_3 = cv2.imread(target_path_4_3)
        target_16_9 = cv2.imread(target_path_16_9)
        if target_mask_path_4_3 is not None:
            target_mask_4_3 = cv2.imread(target_mask_path_4_3, cv2.IMREAD_GRAYSCALE)
        else:
            target_mask_4_3 = None
        if target_mask_path_16_9 is not None:
            target_mask_16_9 = cv2.imread(target_mask_path_16_9, cv2.IMREAD_GRAYSCALE)
        else:
            target_mask_16_9 = None

        print(f"Target name 4/3: {target_name_4_3}")
        print(f"Target name 16/9: {target_name_16_9}")
    else:
        target_name = os.path.splitext(os.path.basename(target_path))[0]
        copyfile(target_path, f"{out_path}/target_{target_name}.jpg")
        target = cv2.imread(target_path)
        heightT, widthT, _ = target.shape

        if target_mask_path is not None:
            target_mask = cv2.imread(target_mask_path, cv2.IMREAD_GRAYSCALE)
        else:
            target_mask = None
        print(f"Target name: {target_name}")

    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)

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
    old_target_path = target_path

    for count, image_path in enumerate(images):
        print("-----------------")
        print(f"Image number {count}/{len(images)}")

        image_name = os.path.splitext(os.path.basename(image_path))[0]
        image = cv2.imread(image_path)
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        height, width, _ = image.shape
        if params.getboolean('Alignment', 'resize_image'):
            image = imutils.resize(image, width=widthT)
        ratio = width/height 

        if old_target_path is None:
            if abs(ratio - 4/3) <= abs(ratio - 16/9):
                target_path = target_path_4_3
                target_name = target_name_4_3
                target = target_4_3
                target_mask = target_mask_4_3
            else:
                target_path = target_path_16_9
                target_name = target_name_16_9
                target = target_16_9
                target_mask = target_mask_16_9

        print(f"Aligning image {image_name}")
    
        method_param = None
        if image_path == target_path:
            print("Target, skipping")
            pass
        else:
            scores[image_name] = {}
            tried = []
            best_method_name = default_method_name
            best_norm_name = default_norm_name
            method_param = get_method_param_value(best_method_name, params)
            try:
                best_aligned, best_indicators = align_and_write(image, image_gray, target, 
                    target_gray, target_mask, image_name, target_name, best_method_name, 
                    best_norm_name, method_param,  out_path_debug, keep_percent, threshold_dist, 
                    use_mask)
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
                "homography_det": best_indicators["homography_det"],
                "homography_norm_value": best_indicators["homography_norm"], 
                "%_covering": best_indicators["percent_covering"],
                "nb_keypoints": best_indicators["nb_keypoints"],
                "mean_dist": best_indicators["mean_dist_between_keypoints"],
                "new_cent_int": best_indicators["projected_center_intensity"],
                "new_cent_loc_ratio": best_indicators["projected_center_location_dist_ratio"]
            })
            retry_alignment = params.getboolean("Retry", "retry_alignment")
            if retry_alignment and \
                (float(best_indicators["homography_norm"]) \
                    >= float(params["Retry"]["homography_norm_max"]) or
                float(best_indicators["percent_covering"]) \
                    <= float(params["Retry"]["percent_covering_min"]) or
                float(best_indicators["mean_dist_between_keypoints"]) \
                    >= float(params["Retry"]["mean_dist_between_keypoints"]) or
                float(best_indicators["projected_center_location_dist_ratio"][0]) \
                    >= float(params["Retry"]["projected_center_location_dist_ratio_max"]) or
                float(best_indicators["projected_center_location_dist_ratio"][1]) \
                    >= float(params["Retry"]["projected_center_location_dist_ratio_max"])):
                print(f"Searching for best alignment method")
                for (method_name, norm_name) in format_parameters_to_try(params):
                    if method_name == default_method_name and norm_name == default_norm_name:
                        continue
                    method_param = get_method_param_value(method_name, params)
                    try:
                        aligned, indicators = align_and_write(image, image_gray, target,
                            target_gray, target_mask, image_name, target_name, method_name,
                            norm_name, method_param, out_path_debug, keep_percent, 
                            threshold_dist, use_mask)
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
                        "homography_det": indicators["homography_det"],
                        "homography_norm_value": indicators["homography_norm"], 
                        "%_covering": indicators["percent_covering"],
                        "nb_keypoints": indicators["nb_keypoints"],
                        "mean_dist": indicators["mean_dist_between_keypoints"],
                        "new_cent_int": indicators["projected_center_intensity"],
                        "new_cent_loc_ratio": indicators["projected_center_location_dist_ratio"]
                    })
                    if indicators["homography_norm"] < best_indicators["homography_norm"]:
                        best_aligned = aligned
                        best_indicators = indicators
   
            if (float(best_indicators["homography_norm"])
                    > float(params["Filter"]["homography_norm_max"]) or
                float(best_indicators["percent_covering"]) 
                    <= float(params["Filter"]["percent_covering_min"]) or
                float(best_indicators["mean_dist_between_keypoints"]) \
                    >= float(params["Filter"]["mean_dist_between_keypoints"]) or
                float(best_indicators["projected_center_location_dist_ratio"][0]) \
                    > float(params["Filter"]["projected_center_location_dist_ratio_max"]) or
                float(best_indicators["projected_center_location_dist_ratio"][1]) \
                    > float(params["Filter"]["projected_center_location_dist_ratio_max"])):
                print(f"Failed to align image {image_name}")
                cv2.imwrite(f"{out_path}/{image_name}_{target_name}_tried_to_align_with_{best_method_name}_{best_norm_name}_.jpg",
                    best_aligned)
                scores[image_name] = {
                    "target":target_name,
                    "aligned": False,
                    "is_fake": False,
                    "method": best_method_name, 
                    "norm": best_norm_name, 
                    "homography_det": best_indicators["homography_det"],
                    "homography_norm_value": best_indicators["homography_norm"], 
                    "%_covering": best_indicators["percent_covering"],
                    "nb_keypoints": best_indicators["nb_keypoints"],
                    "mean_dist":  best_indicators["mean_dist_between_keypoints"], 
                    "new_cent_int": best_indicators["projected_center_intensity"],
                    "new_cent_loc_ratio": best_indicators["projected_center_location_dist_ratio"]
                }
            else:
                print(f"Image {image_name} aligned with {best_method_name}, "\
                  f"{best_norm_name} with score {best_indicators['homography_norm']}")
                cv2.imwrite(f"{out_path}/{image_name}_{target_name}_aligned_with_{best_method_name}_{best_norm_name}_.jpg",
                    best_aligned)
                scores[image_name] = {
                    "target": target_name,
                    "aligned": True,
                    "is_fake": False,
                    "method": best_method_name, 
                    "norm": best_norm_name,
                    "homography_det": best_indicators["homography_det"],
                    "homography_norm_value": best_indicators["homography_norm"], 
                    "%_covering": best_indicators["percent_covering"],
                    "nb_keypoints": best_indicators["nb_keypoints"],
                    "mean_dist":  best_indicators["mean_dist_between_keypoints"], 
                    "new_cent_int": best_indicators["projected_center_intensity"],
                    "new_cent_loc_ratio": best_indicators["projected_center_location_dist_ratio"]
                }

            if len(tried) > 0:
                scores[image_name].update({"tried": tried})
            aligned_ov = best_aligned.copy()
            cv2.addWeighted(target, 0.5, best_aligned, 0.5, 0, aligned_ov)
            cv2.imwrite(f"{out_path}/{image_name}_{target_name}_{best_method_name}_{best_norm_name}_overlay.jpg", aligned_ov)
            with open(out_path + '/scores.json', 'w', encoding="utf-8") as outfile:
                json.dump(scores, outfile, indent=4)

    export_stats(out_path + '/scores.json', start_time)
    

if __name__ == "__main__":
    batch_test()
