import cv2
import numpy as np
import os
import json

dim_ratio = -1


def find_coo(frames_path):
    markup_path = ""
    # get the homography coordinates from the json file
    if 'origins' in frames_path:
        markup_path = "E:/dataset MIDV HOLO/markup/origins"
    elif 'copy_without_holo' in frames_path:
        markup_path = "E:/dataset MIDV HOLO/markup/fraud/copy_without_holo"
    elif 'photo_holo_copy' in frames_path:
        markup_path = "E:/dataset MIDV HOLO/markup/fraud/photo_holo_copy"
    elif 'photo_replacement' in frames_path:
        markup_path = "E:/dataset MIDV HOLO/markup/fraud/photo_replacement"
    elif 'pseudo_holo_copy' in frames_path:
        markup_path = "E:/dataset MIDV HOLO/markup/fraud/pseudo_holo_copy"
    # Ratio of iD and passport image
    psp_dim_ratio = 4920 / 3463
    ID_dim_ratio = 3370 / 2127
    global dim_ratio
    nbr_of_markups = 0
    # getting the folder name
    if "\\" in frames_path :
        splted_path = frames_path.split('\\')
        folder_name = [i for i in splted_path if 'psp' in i][0]
    else :
        splted_path = frames_path.split('/')
        folder_name = [i for i in splted_path if 'psp' in i][0]
    # find the folder and get the coordinates
    coordinates_dict = {}  # <- list of homographie coordinates for each frame
    for root, dirs, files in os.walk(markup_path):
        # Check if the current folder is "psp01_01_01"
        if os.path.basename(root) in [folder_name]:
            nbr_of_markups = len(files)
            for file in files:
                f = open(root + '/' + file)
                data = json.load(f)
                # get document type
                doc_type = data['document']['document_type']
                if 'passport' in doc_type:
                    doc_type_complete = doc_type + ':main'
                    dim_ratio = psp_dim_ratio
                else:  # ID
                    doc_type_complete = doc_type + ':front'
                    dim_ratio = ID_dim_ratio
                template_quad = data['document']['templates'][doc_type_complete]['template_quad']
                coordinates_dict[file] = template_quad

    return coordinates_dict


def Homo(coordinates_dict, path, files):
    y_size = 400  # to set manually
    x_size = int(y_size * dim_ratio)
    sorted_dict = dict(sorted(coordinates_dict.items(), key=lambda item: item[0]))
    keys_list = list(sorted_dict.keys())
    cpt = 0
    images_homo = []
    for file in files:
        if 'jpg' in file :
            frame = cv2.imread(path + "/" + file)
            # Read source image.
            im_src = frame
            # Four corners of the passport in source image
            pts_src = np.array(coordinates_dict[keys_list[cpt]])

            # Four corners of the passport in destination image.
            pts_dst = np.array([[0, 0], [x_size - 1, 0], [x_size - 1, y_size - 1], [0, y_size - 1]])

            # Calculate Homography
            h, status = cv2.findHomography(pts_src, pts_dst)

            # Warp source image to destination based on homography
            im_out = cv2.warpPerspective(im_src, h, (x_size, y_size))
            images_homo.append(im_out)
            cpt += 1
    return images_homo


def all_(mother_frames_path):
    cpt_glob = 0
    for current_path, directories, files in os.walk(mother_frames_path):
        if files != [] and 'psp' in current_path and "photo_replacement" not in current_path:
            print(cpt_glob)
            print(current_path)
            coordinates_dict = find_coo(current_path)
            Homos = Homo(coordinates_dict, current_path,files)
            directory = current_path.replace('images', 'images_homo')
            os.makedirs(directory, exist_ok=True)
            cpt = 1
            for homo in Homos:
                cv2.imwrite(directory + "/img_" + str(cpt).zfill(4) + '.jpg', homo)
                cpt += 1
            cpt_glob += 1


if __name__ == '__main__':
    all_("E:/dataset MIDV HOLO/images/")

