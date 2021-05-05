from pyimagesearch.alignment import align_images
import numpy as np
import argparse
import imutils
import cv2
import os


def align_images(image, template, method="surf", norm=cv2.NORM_L2, maxFeatures=500, keepPercent=0.2,
    debug=False):
    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    templateGray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    if method == "orb":
        orb = cv2.ORB_create(maxFeatures)
        (kpsA, descsA) = orb.detectAndCompute(imageGray, None)
        (kpsB, descsB) = orb.detectAndCompute(templateGray, None)
        norm = cv2.NORM_HAMMING
    elif method == "sift":
        sift = cv2.xfeatures2d.SIFT_create()
        kpsA, descsA = sift.detectAndCompute(imageGray, None)
        kpsB, descsB = sift.detectAndCompute(templateGray, None)
    elif method == "surf":
        surf = cv2.xfeatures2d.SURF_create()
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

    # check to see if we should visualize the matched keypoints
    if debug:
        matchedVis = cv2.drawMatches(image, kpsA, template, kpsB,
            matches, None)
        matchedVis = imutils.resize(matchedVis, width=1000)
        cv2.imshow("Matched Keypoints", matchedVis)
        cv2.waitKey(0)

    # Allocate memory for the keypoints (x,y-coordinates) from the
    # top matches -- we'll use these coordinates to compute our
    # homography matrix
    ptsA = np.zeros((len(matches), 2), dtype="float")
    ptsB = np.zeros((len(matches), 2), dtype="float")

    # Loop over the top matches
    for (i, m) in enumerate(matches):
        # Indicate that the two keypoints in the respective images
        #   map to each other
        ptsA[i] = kpsA[m.queryIdx].pt
        ptsB[i] = kpsB[m.trainIdx].pt

    # Compute the homography matrix between the two sets of matched
    #   points
    (H, mask) = cv2.findHomography(ptsA, ptsB, method=cv2.RANSAC)

    # Use the homography matrix to align the images
    (h, w) = template.shape[:2]
    aligned = cv2.warpPerspective(image, H, (w, h))

    return aligned

def main():
    # image = cv2.imread(args["image"])
    # template = cv2.imread(args["template"])
    # image = cv2.imread("./images/tasmani_coastsnap1.jpg")
    # template = cv2.imread("./images/tasmani_coastsnap2.jpg")

    # image_path = "./images/tasmani_coastsnap3.jpg"
    # template_path = "./images/tasmani_coastsnap4.jpg"

    # image_path = "./images/narrabeen_coastsnap2.jpg"
    # template_path = "./images/narrabeen_coastsnap3.jpg"
    image_path = "./images/palombaggia1.jpg"
    template_path = "./images/palombaggia2.jpg"

    # image_path = "./images/narrabeen_coastsnap2.jpg"
    # template_path = "./images/narrabeen_coastsnap3.jpg"
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    template_name = os.path.splitext(os.path.basename(template_path))[0]
    image = cv2.imread(image_path)
    template = cv2.imread(template_path)

    method_name = "sift"
    norm_name = "L2"
    if norm_name == "L1":
        norm=cv2.NORM_L1
    if norm_name == "L2":
        norm=cv2.NORM_L2

    print("[INFO] aligning images...")
    aligned = align_images(image, template, method=method_name, norm=norm, debug=False)

    image = imutils.resize(image, width=700)
    aligned = imutils.resize(aligned, width=700)

    # cv2.imshow("Aligned", aligned)
    template = imutils.resize(template, width=700)

    stacked = np.hstack([aligned, template])

    cv2.imwrite(f"{image_name}.jpg", image)
    cv2.imwrite(f"{template_name}.jpg", template)
    cv2.imwrite(f"{image_name}_{template_name}_{method_name}_{norm_name}_stacked.jpg", stacked)

    overlay = template.copy()
    output = aligned.copy()
    cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)
    cv2.imwrite(f"{image_name}_{template_name}_{method_name}_{norm_name}_overlay.jpg", output)

    # cv2.imshow("Image Alignment Stacked", stacked)
    # cv2.imshow("Image Alignment Overlay", output)
    cv2.waitKey(0)


if __name__ == "__main__":
    main()
