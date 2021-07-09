import numpy as np
import argparse
import imutils
import cv2
import os
import math
from numpy import linalg as LA
import json
import glob

def align_images(image, template, method="SIFT", norm_name="L2", keepPercent=0.3, debug=False):
    
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    if norm_name == "L1":
        norm=cv2.NORM_L1
    if norm_name == "L2":
        norm=cv2.NORM_L2

    if method == "ORB":
        orb = cv2.ORB_create()
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

    bf = cv2.BFMatcher(norm, crossCheck=True)
    matches = bf.match(descsA, descsB)	

    # Sort the matches by their distance (the smaller the distance,
    #   the "more similar" the features are)
    matches = sorted(matches, key=lambda x:x.distance)

    # Keep only the top matches
    keep = int(len(matches) * keepPercent)
    matches = matches[:keep]

    # print(f"Number of keypoints: {len(matches)}")
    matchedVis = cv2.drawMatches(image, kpsA, template, kpsB, matches, None)
    matchedVis = imutils.resize(matchedVis, width=1900)

    # Allocate memory for the keypoints (x,y-coordinates) from the top matches
    #   -- we'll use these coordinates to compute our homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    # Loop over the top matches
    for (i, m) in enumerate(matches):
        # Indicate that the two keypoints in the respective images  map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # Compute the homography matrix between the two sets of matched points
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

    homo_norm = LA.norm(H)
    print(f"Homography matrix norm: {homo_norm}")
    # Use the homography matrix to align the images
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))

    return aligned, matchedVis, homo_norm


def batch_test(debug):
    folder_path = "./images/Lacanau_Kayok_VueNord"
    # folder_path = "./images/Lacanau_Kayok_VueNord (copie)"
    # folder_path = "./images/SaintJeanDeLuz_Lafitenia_VueNord"
    # folder_path = "./images/Capbreton_Santocha_VueSud"
    out_path = folder_path.replace("./images", "./results")
    try:
        os.makedirs(out_path)    
        print("Directory " , out_path ,  " Created ")
    except FileExistsError:
        pass
    
    template_path = "./images/Lacanau_Kayok_VueNord/20201204_144234.jpg"
    # template_path = "./images/SaintJeanDeLuz_Lafitenia_VueNord/20210308_111406.jpg"
    # template_path = "./images/Capbreton_Santocha_VueSud/IMG_20210409_101128.jpg"
    template_name = os.path.splitext(os.path.basename(template_path))[0]
    template = cv2.imread(template_path)
    template_min = imutils.resize(template, width=950)
    print(f"Template name {template_name}")

    scores = {}
    for image_path in glob.glob(folder_path +'/*.jpg'):
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        print("-----------------")
        print(f"Aligning image {image_name}")
    
        if image_path == template_path:
            print("Template, skipping")
            pass
        else:
            scores[image_name] = {}
            image = cv2.imread(image_path)
            best_method_name = "SIFT"
            best_norm_name = "L2"
            best_aligned, best_homo_norm = align_and_write(image, template, template_min, image_name, template_name, best_method_name, best_norm_name, out_path, debug)
            if best_homo_norm >= 200:
                print(f"Searching for best alignment method")
                for (method_name, norm_name) in [("SURF", "L1"), ("SURF", "L2"), ("SIFT", "L1")]:
                    aligned, homo_norm = align_and_write(image, template, template_min, image_name, template_name, method_name, norm_name, out_path, debug)
                    if homo_norm < best_homo_norm:
                        best_homo_norm = homo_norm
                        best_aligned = aligned
                        best_method_name = method_name
                        best_norm_name = norm_name
            if best_homo_norm < 600:
                print(f"Image {image_name} aligned with {best_method_name}, {best_norm_name} with score {best_homo_norm}")
                cv2.imwrite(f"{out_path}/{image_name}_{template_name}_aligned.jpg", best_aligned)
                scores[image_name] = {
                    "success":True, 
                    "homography_norm_value": best_homo_norm, 
                    "method": best_method_name, 
                    "norm":best_norm_name, 
                    "template":template_name
                }
            else:
                print(f"Failed to align image {image_name}...")
                scores[image_name] = {
                    "success":False, 
                    "homography_norm_value": best_homo_norm, 
                    "template":template_name
                }
            with open(out_path + '/scores.json', 'w') as outfile:
                json.dump(scores, outfile)

    with open(out_path + '/scores.json') as json_file:
        scores = json.load(json_file)

        values = []
        for item in scores.items():
            print(item)
            values.append(item[1]["homography_norm_value"])
        scores["stats"] = {
            "min": np.min(values),
            "max": np.max(values),
            "mean": np.mean(values),
            "median": np.median(values)
        }
        with open(out_path + '/scores.json', 'w') as outfile:
            json.dump(scores, outfile)


def align_and_write(image, template, template_min, image_name, template_name, method_name, norm_name, out_path, debug):
    debug_image_name = f"{image_name}_{template_name}_{method_name}_{norm_name}_descriptors_matching.jpg"

    aligned, matchedVis, homo_norm = align_images(image, template, method=method_name, norm_name=norm_name, debug=debug)
    cv2.imwrite(f"{out_path}/{image_name}_{template_name}_{method_name}_{norm_name}_aligned.jpg", aligned)
    cv2.imwrite(f"{out_path}/{image_name}.jpg", image)
    cv2.imwrite(f"{out_path}/template_{template_name}.jpg", template)
    cv2.imwrite(f"{out_path}/{debug_image_name}", matchedVis)

    aligned_min = imutils.resize(aligned, width=950)

    stacked = np.hstack([aligned_min, template_min])

    cv2.imwrite(f"{out_path}/{image_name}_{template_name}_{method_name}_{norm_name}_stacked.jpg", stacked)

    cv2.addWeighted(template_min, 0.5, aligned_min, 0.5, 0, aligned_min)
    cv2.imwrite(f"{out_path}/{image_name}_{template_name}_{method_name}_{norm_name}_overlay.jpg", aligned_min)

    return aligned, homo_norm



def one_test(debug):
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

    
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    template_name = os.path.splitext(os.path.basename(template_path))[0]
    print(f"input: {image_name}, output: {template_name}")
    image = cv2.imread(image_path)
    template = cv2.imread(template_path)
    
    method_name = "SIFT"
    norm_name = "L2"
    debug_image_name = f"{image_name}_{template_name}_{method_name}_{norm_name}_descriptors_matching.jpg"

    print("[INFO] aligning images...")
    aligned, matchedVis, homo_norm = align_images(image, template, method=method_name, norm_name=norm_name, debug=debug)
    cv2.imwrite(f"results/{image_name}_aligned.jpg", aligned)
    cv2.imwrite(f"results/{image_name}.jpg", image)
    cv2.imwrite(f"results/template_{template_name}.jpg", template)
    cv2.imwrite(f"results/{debug_image_name}", matchedVis)
    
    image = imutils.resize(image, width=950)
    aligned = imutils.resize(aligned, width=950)
    template = imutils.resize(template, width=950)

    stacked = np.hstack([aligned, template])

    cv2.imwrite(f"results/{image_name}_{template_name}_{method_name}_{norm_name}_stacked.jpg", stacked)

    overlay = template.copy()
    output = aligned.copy()
    cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
    cv2.imwrite(f"results/{image_name}_{template_name}_{method_name}_{norm_name}_overlay.jpg", output)



def main():
    # one_test(True)
    batch_test(True)


if __name__ == "__main__":
    main()
