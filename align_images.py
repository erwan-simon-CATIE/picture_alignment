# coding: utf-8

import numpy as np
import argparse
import imutils
import cv2
import os
import math
from numpy import linalg as LA
import json
import glob
from shutil import copyfile
import time


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
        homo_norm = LA.norm(H)
    except TypeError as e:
        print("Exception occured, norm error:", e)
        raise ValueError("Norm inapplicable")

    print(f"Homography matrix norm: {homo_norm}")
    # Use the homography matrix to align the images
    
    white = np.zeros([heightA, widthA, 1],dtype=np.uint8)
    white.fill(255)
    aligned = cv2.warpPerspective(image, H, (widthB, heightB))
    aligned_mask = cv2.warpPerspective(white, H, (widthB, heightB))
    n_white_pix = np.sum(aligned_mask == 255)
    per_white_pix = 100 * n_white_pix / (heightB * widthB)
    print(f"% of covering: {per_white_pix}")
    print(f"Keypoints n°: {nb_keypoints}")
    print(f"Mean distance: {mean_dist}")

    return aligned, homo_norm, per_white_pix, nb_keypoints, mean_dist, matchedVis


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
    # folder_path = "./images/Manly"
    # folder_path = "./images/North_Narrabeen"
    # folder_path = "./images/test_rapide4"
    folder_path = "./images/test_coastsnap"
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
    # template_path = "./images/Manly/tp9pzlhrd0pfdwyfpptj6czxmiaq8554.jpg"

    template_path = None
    template_path_4_3 = "./images/North_Narrabeen/4gf0l6xp79st2bukc739xhzqw5tchopv.jpg"
    template_path_16_9 = "./images/North_Narrabeen/1m81bw22qltgx1bj1y8pfsy1a8gym72y.jpg"
    
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

    scores = {}
    images = sorted(glob.glob(folder_path +'/*.jpg'))
    nb_images = len(images)
    print(f"{nb_images} images found.")
    old_template_path = template_path

    for count, image_path in enumerate(images):
        if count < 256: #fixme
            continue
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
                best_aligned, best_homo_norm, best_covering, best_nb_keypoints, best_mean_dist = align_and_write(image, template, image_name, template_name, best_method_name, best_norm_name, out_path_debug, threshold_dist)
            except ValueError as e:
                print("Exception occured:", e)
                scores[image_name].update({"aligned": False, "is_fake": True, "exception": str(e)})
                with open(out_path + '/scores.json', 'w') as outfile:
                    json.dump(scores, outfile, indent=4)
                continue

            tried.append({
                        "homography_norm_value": best_homo_norm, 
                        "%covering": best_covering,
                        "nb_keypoints": best_nb_keypoints,
                        "mean_dist": best_mean_dist,
                        "method": best_method_name, 
                        "norm": best_norm_name
                    })
            if best_homo_norm >= 150 or best_covering <= 90 or best_mean_dist <= 0.35:
                print(f"Searching for best alignment method")
                for (method_name, norm_name) in [("SURF", "L1"), ("SURF", "L2"), ("SIFT", "L1"), ("SIFT", "L2"), ("BRISK", "HAMMING")]:
                    if method_name == default_method_name and norm_name == default_norm_name:
                        continue
                    try:
                        aligned, homo_norm, covering, nb_keypoints, mean_dist = align_and_write(image, template, image_name, template_name, method_name, norm_name, out_path_debug, threshold_dist)
                    except ValueError as e:
                        print("Exception occured:", e)
                        scores[image_name].update({"aligned":False, "is_fake": True, "exception": str(e)})
                        if len(tried) > 0:
                            scores[image_name].update({"tried": tried})
                        with open(out_path + '/scores.json', 'w') as outfile:
                            json.dump(scores, outfile, indent=4)
                        continue
                    tried.append({
                        "homography_norm_value": homo_norm, 
                        "%covering": covering,
                        "nb_keypoints": nb_keypoints,
                        "mean_dist": mean_dist,
                        "method": method_name, 
                        "norm": norm_name
                    })
                    if homo_norm < best_homo_norm:
                        best_homo_norm = homo_norm
                        best_aligned = aligned
                        best_method_name = method_name
                        best_norm_name = norm_name
                        best_covering = covering
                        best_nb_keypoints = nb_keypoints
                        best_mean_dist = mean_dist
            if best_homo_norm > 300 or best_covering < 85 or best_mean_dist > 0.35:
                # Try to realigned the aligned image with the template
                try:
                    realigned, homo_norm_re, covering_re, nb_keypoints_re, mean_dist_re = align_and_write(aligned, image, image_name, template_name, best_method_name, best_norm_name, out_path_debug, threshold_dist)
                except ValueError as e:
                        print("Exception occured:", e)
                        realigned = None
                        homo_norm_re = 9999
                        covering_re = 0
                        nb_keypoints_re = 0
                        mean_dist_re = 100
                tried_realignment = True
                print(f"Realigned with homo_norm_re {homo_norm_re}, covering_re {covering_re}, nb_keypoints_re {nb_keypoints_re}, mean_dist_re {mean_dist_re}")
                cv2.imwrite(f"{out_path_debug}/{image_name}_{template_name}_realigned.jpg", realigned)
                if homo_norm_re > 700 or mean_dist_re >= 0.6:
                    print(f"Image {image_name} is not a photo of the template beach...")
                    scores[image_name] = {
                        "aligned": False,
                        "is_fake": True, 
                        "homography_norm_value": best_homo_norm, 
                        "%covering": best_covering,
                        "nb_keypoints": best_nb_keypoints,
                        "mean_dist": best_mean_dist,
                        "template": template_name,
                        "realignment": {
                            "homography_norm_value_re": homo_norm_re, 
                            "%covering_re": covering_re,
                            "nb_keypoints_re": nb_keypoints_re,
                            "mean_dist_re": mean_dist_re
                        }
                    }
                    if len(tried) > 0:
                        scores[image_name].update({"tried": tried})
                    with open(out_path + '/scores.json', 'w') as outfile:
                        json.dump(scores, outfile, indent=4)
                    continue

            if best_homo_norm > 500 or best_covering <= 75 or best_mean_dist >= 0.5:
                print(f"Failed to align image {image_name}, but nevertheless, it seems to be the right beach image")
                cv2.imwrite(f"{out_path}/{image_name}_{template_name}_tried_to_be_aligned_with_{best_method_name}_{best_norm_name}_.jpg", best_aligned)
                scores[image_name] = {
                    "aligned": False,
                    "is_fake": False,
                    "homography_norm_value": best_homo_norm, 
                    "%covering": best_covering,
                    "nb_keypoints": best_nb_keypoints,
                    "mean_dist": best_mean_dist,
                    "method": best_method_name, 
                    "norm": best_norm_name, 
                    "template":template_name
                }
                if len(tried) > 0:
                    scores[image_name].update({"tried": tried})
                if tried_realignment:
                    scores[image_name].update({
                        "realignment": {
                            "homography_norm_value_re": homo_norm_re, 
                            "%covering_re": covering_re,
                            "nb_keypoints_re": nb_keypoints_re,
                            "mean_dist_re": mean_dist_re
                        }})
            else:
                print(f"Image {image_name} aligned with {best_method_name}, {best_norm_name} with score {best_homo_norm}")
                cv2.imwrite(f"{out_path}/{image_name}_{template_name}_aligned_with_{best_method_name}_{best_norm_name}_.jpg", best_aligned)
                scores[image_name] = {
                    "aligned": True,
                    "is_fake": False,
                    "homography_norm_value": best_homo_norm, 
                    "%covering": best_covering,
                    "nb_keypoints": best_nb_keypoints,
                    "mean_dist": best_mean_dist,
                    "method": best_method_name, 
                    "norm": best_norm_name, 
                    "template": template_name
                }
                if len(tried) > 0:
                    scores[image_name].update({"tried": tried})
                if tried_realignment:
                    scores[image_name].update({
                        "realignment": {
                            "homography_norm_value_re": homo_norm_re, 
                            "%covering_re": covering_re,
                            "nb_keypoints_re": nb_keypoints_re,
                            "mean_dist_re": mean_dist_re
                        }})
            aligned_ov = best_aligned.copy()
            cv2.addWeighted(template, 0.5, best_aligned, 0.5, 0, aligned_ov)
            cv2.imwrite(f"{out_path}/{image_name}_{template_name}_{best_method_name}_{best_norm_name}_overlay.jpg", aligned_ov)
            with open(out_path + '/scores.json', 'w') as outfile:
                json.dump(scores, outfile, indent=4)
    if os.path.isfile(out_path + '/scores.json'):
        try:
            with open(out_path + '/scores.json') as json_file:
                scores = json.load(json_file)
                values_homo = []
                values_cov = []
                values_keypoints = []
                values_mean_dist = []
                for item in scores.items():
                    try:
                        print(item)
                        values_homo.append(item[1]["homography_norm_value"])
                        values_cov.append(item[1]["%covering"])
                        values_keypoints.append(item[1]["nb_keypoints"])
                        values_mean_dist.append(item[1]["mean_dist"])
                    except KeyError as e:
                        print("Exception occured, Key error:", e)
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
                    scores["stats"]["%covering"] = {
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
                scores["stats"].update({
                    "comp_time_minutes": (time.time() - start_time) / 60
                })
                with open(out_path + '/scores.json', 'w') as outfile:
                    json.dump(scores, outfile, indent=4)
        except FileNotFoundError as e:
            print("Exception occured:", e)


def align_and_write(image, template, image_name, template_name, method_name, norm_name, out_path_debug, threshold_dist):
    debug_image_name = f"{image_name}_{template_name}_{method_name}_{norm_name}_descriptors_matching.jpg"

    aligned, homo_norm, covering, nb_keypoints, mean_dist, matchedVis = align_images(image, template, method=method_name, norm_name=norm_name, threshold_dist=threshold_dist)
    cv2.imwrite(f"{out_path_debug}/{debug_image_name}", matchedVis)

    aligned_ov = aligned.copy()
    cv2.addWeighted(template, 0.5, aligned, 0.5, 0, aligned_ov)
    cv2.imwrite(f"{out_path_debug}/{image_name}_{template_name}_{method_name}_{norm_name}_overlay.jpg", aligned_ov)
    
    return aligned, homo_norm, covering, nb_keypoints, mean_dist



def one_test():
    out_path = "./results"
    # image = cv2.imread(args["image"])
    # template = cv2.imread(args["template"])
    
    # image_path = "./images/Lacanau_Kayok_Vue_Nord_Reduit/20201215_140836.jpg"
    # template_path = "./images/Lacanau_Kayok_Vue_Nord_Reduit/20201204_144234.jpg"

    # image_path = "./images/palombaggia1.jpg"
    # template_path = "./images/palombaggia2.jpg"
    # image_path = "./exemples/compliqué1/5/tasmani_coastsnap1.jpg"
    # template_path = "./exemples/compliqué1/5/tasmani_coastsnap2.jpg"

    # image_path = "./exemples/compliqué1/4/tasmani_coastsnap1.jpg"
    # template_path = "./exemples/compliqué1/4/tasmani_coastsnap2.jpg"

    # image_path = "./exemples/trop_dur/9/bresil_coastsnap1.jpg"
    # template_path = "./exemples/trop_dur/9/bresil_coastsnap2.jpg"

    # No match example
    # image_path = "./images/SaintJeanDeLuz_Lafitenia_VueNord/20210308_111406.jpg"
    # template_path = "./images/Lacanau_Kayok_Vue_Nord_Reduit/20201204_144234.jpg"

    # image_path = "./images/Capbreton_Santocha_VueSud/IMG_20210409_101128.jpg"
    # template_path = "./images/Capbreton_Santocha_VueSud/IMG_20210409_101206.jpg"

    # FIXME: Plante
    # image_path = "./images/other/blanc.png"
    # template_path = "./images/Lacanau_Kayok_Vue_Nord_Reduit/20201204_144234.jpg"
    
    # Sift marche pas mais surf oui 
    # image_path = "./images/Lacanau_Kayok_VueNord/20201118_124741.jpg"
    # template_path = "./images/Lacanau_Kayok_VueNord/20201204_144234.jpg"

    # Trop dur
    # image_path = "./images/Lacanau_Kayok_VueNord/20201214_172225.jpg"
    # template_path = "./images/Lacanau_Kayok_VueNord/20201204_144234.jpg"

    image_path = "./images/North_Narrabeen/biyg6w29eq4fd95rit4ht4kndzprfrh4.jpg"
    template_path = None
    template_path_4_3 = "./images/North_Narrabeen/4gf0l6xp79st2bukc739xhzqw5tchopv.jpg"
    template_path_16_9 = "./images/North_Narrabeen/1m81bw22qltgx1bj1y8pfsy1a8gym72y.jpg"
    
    # BRISK meilleur perf alors que... =/
    # image_path = "./images/North_Narrabeen/cmf39ffgr203d19qysvwjaq3r56hss68.jpg"

    image_name = os.path.splitext(os.path.basename(image_path))[0]
    image = cv2.imread(image_path)
    height, width, channels = image.shape
    ratio = width/height 

    if template_path is None:
        if abs(ratio - 4/3) <= abs(ratio - 16/9):
            template_path = template_path_4_3
        else:
            template_path = template_path_16_9

    template_name = os.path.splitext(os.path.basename(template_path))[0]
    copyfile(template_path, f"{out_path}/template_{template_name}.jpg")
    copyfile(image_path, f"{out_path}/{image_name}.jpg")

    print(f"input: {image_name}, output: {template_name}")
    template = cv2.imread(template_path)
    
    method_name = "SIFT"
    norm_name = "L2"
    threshold_dist = 1.5
    debug_image_name = f"{image_name}_{template_name}_{method_name}_{norm_name}_descriptors_matching.jpg"

    print("[INFO] aligning images...")
    aligned, homo_norm, covering, nb_keypoints, mean_dist, matchedVis = align_images(image, template, method=method_name, norm_name=norm_name, threshold_dist=threshold_dist)
    cv2.imwrite(f"results/{image_name}_aligned.jpg", aligned)
    cv2.imwrite(f"results/{debug_image_name}", matchedVis)

    overlay = template.copy()
    output = aligned.copy()
    cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
    cv2.imwritemplate_namete(f"results/{image_name}_{template_name}_{method_name}_{norm_name}_overlay.jpg", output)



def main():
    # one_test()
    batch_test()


if __name__ == "__main__":
    main()
