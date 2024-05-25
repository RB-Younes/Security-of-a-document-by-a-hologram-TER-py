import os
import shutil
import sys


def copy_images(source_dir, dest_dir):
    for filename in os.listdir(source_dir):
        source_file = os.path.join(source_dir, filename)
        if os.path.isfile(source_file):
            shutil.copy(source_file, dest_dir)


def restructure_folders(main_dir, dest_dir):
    for root, dirs, files in os.walk(main_dir):

        for dirname in dirs:
            if dirname.startswith("psp_"):
                parent_folder = os.path.basename(os.path.dirname(root))
                if parent_folder in ["train", "test", "validation"]:

                    holo_dir = os.path.join(dest_dir, parent_folder, "Holo")
                    no_holo_dir = os.path.join(dest_dir, parent_folder, "No-Holo")
                    if not os.path.exists(holo_dir):
                        os.makedirs(holo_dir)
                    if not os.path.exists(no_holo_dir):
                        os.makedirs(no_holo_dir)
                    if "No-Holo" in root:
                        copy_images(os.path.join(root, dirname), no_holo_dir)
                    elif "Holo" in root:
                        copy_images(os.path.join(root, dirname), holo_dir)


if __name__ == "__main__":
    main_directory = "E:/dataset MIDV HOLO/test_pho_rep_80/"
    final_directory = "E:/dataset MIDV HOLO/test_photo_rep_final_80/"

    restructure_folders(main_directory, final_directory)
