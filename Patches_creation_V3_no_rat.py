import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import json

psp_width = 600
psp_height = 400

patch_height = 40
patch_width = 40
patch_size = (patch_height,patch_width)  # [1, 2, 4, 8, 71, 142, 284, 568]

num_rows = 9
num_cols = 9

number_of_patch = int((psp_height / patch_height) * (psp_width / patch_width))


def get_patch(patch_id, image_path):  # get only one patch withe specified ID

    image = cv2.imread(image_path)
    height, width = image.shape[:2]

    cpt = 0
    for y in range(0, height, patch_height):
        for x in range(0, width, patch_width):
            if cpt == patch_id:
                return image[y:y + patch_height, x:x + patch_width]
            else:
                cpt += 1


# get the mosaic of the specified patch
def get_mosaic(patch_id, frames_path):
    mosa = []
    for root, dirs, files in os.walk(frames_path):
        for file in files:
            mosa.append(get_patch(patch_id, root + '/' + file))

    return mosa


def get_patches_mask(patches_mask):
    labels = []
    for i, patch in enumerate(patches_mask):
        if np.all(patch == 0):
            labels.append('No-Holo')
        else:
            labels.append('Holo')
    return labels


def get_all_mosaics(current_path, labels_mask):
    mosaics = {'Holo': [], 'No-Holo': []}
    for k in range(number_of_patch):
        mosa = get_mosaic(k, current_path)
        patches_array = np.array(mosa)

        final_image = np.zeros((num_rows * patch_height, num_cols * patch_width, 3), dtype=np.uint8)

        for i in range(min(num_rows, len(mosa))):
            for j in range(min(num_cols, len(mosa))):
                patch_index = i * num_cols + j
                if patch_index < len(mosa):
                    final_image[i * patch_height: (i + 1) * patch_height, j * patch_width: (j + 1) * patch_width, :] = \
                        patches_array[patch_index]
                else:
                    final_image[i * patch_height: (i + 1) * patch_height, j * patch_width: (j + 1) * patch_width, :] = 0

        mosaics[labels_mask[k]].append(final_image)
    return mosaics


def divide_image_into_patches(image_path):
    image = cv2.imread(image_path)
    height, width = image.shape[:2]
    patch_height, patch_width = patch_size
    patches = []
    cpt = 0
    for y in range(0, height, patch_height):
        for x in range(0, width, patch_width):
            patch = image[y:y + patch_height, x:x + patch_width]
            patches.append(patch)
            cpt += 1
    return patches, cpt


def all_(mother_frames_path, save_path):
    cpt_glob = 0
    # get the labels of the images
    patches_mask, num_patches = divide_image_into_patches("passport_hologram_mask_600_400.png")
    labels_mask = get_patches_mask(patches_mask)
    os.makedirs(save_path + "/Holo", exist_ok=True)
    os.makedirs(save_path + "/No-Holo", exist_ok=True)
    cpt = 0
    for current_path, directories, files in os.walk(mother_frames_path):
        if files != [] and 'psp' in current_path:
            print(cpt_glob)
            print(current_path)
            if 'fraud' not in current_path:
                result = get_all_mosaics(current_path, labels_mask)
                for key, val in result.items():
                    if key == "Holo":
                        os.makedirs(save_path + "/Holo" + "/psp_" + str(cpt_glob).zfill(5), exist_ok=True)
                        for img in val:
                            cv2.imwrite(save_path + "/Holo" + "/psp_" + str(cpt_glob).zfill(5) + "/mosa_" + str(cpt).zfill(5) + '.jpg', img)
                            cpt += 1
                    else:
                        os.makedirs(save_path + "/No-Holo" + "/psp_" + str(cpt_glob).zfill(5), exist_ok=True)
                        for img in val:
                            cv2.imwrite(save_path + "/No-Holo" + "/psp_" + str(cpt_glob).zfill(5) + "/mosa_" + str(cpt).zfill(5) + '.jpg', img)
                            cpt += 1
            else:
                result = get_all_mosaics(current_path, labels_mask)
                os.makedirs(save_path + "/No-Holo" + "/psp_" + str(cpt_glob).zfill(5), exist_ok=True)
                for key, val in result.items():
                    for img in val:
                        cv2.imwrite(save_path + "/No-Holo" + "/psp_" + str(cpt_glob).zfill(5) + "/mosa_" + str(cpt).zfill(5) + '.jpg', img)
                        cpt += 1
            cpt_glob += 1


if __name__ == '__main__':
    #
    all_("E:/dataset MIDV HOLO/images_homo_no_rat", "E:/dataset MIDV HOLO/Mosaics_V3_no_rat")
