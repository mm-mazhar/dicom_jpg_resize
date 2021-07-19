# Run in console
#!conda install gdcm -c conda-forge -y

import os
import sys
import numpy as np
import pandas as pd
import pydicom
from PIL import Image
from tqdm.auto import tqdm
from pydicom.pixel_data_handlers.util import apply_voi_lut


def read_xray(path, voi_lut=True, fix_monochrome=True):
    # Original from: https://www.kaggle.com/raddar/convert-dicom-to-np-array-the-correct-way
    dicom = pydicom.read_file(path)

    # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to
    # "human-friendly" view
    if voi_lut:
        img_array = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        img_array = dicom.pixel_array

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        img_array = np.amax(img_array) - img_array

    img_array = img_array - np.min(img_array)
    img_array = img_array / np.max(img_array)
    img_array = (img_array * 255).astype(np.uint8)

    return img_array


def resize(img_array, width, height, keep_ratio=False, resample=Image.LANCZOS):
    # Original from: https://www.kaggle.com/xhlulu/vinbigdata-process-and-resize-to-image
    img = Image.fromarray(img_array)

    if keep_ratio:
        img.thumbnail((width, height))  # , resample)
    else:
        img = img.resize((width, height))  # , resample)

    return img


def main():
    # input_path = sys.argv[1]
    input_path = "C:/Users/mazhar/data/"
    image_id = []
    width = 224
    height = 224
    #   width = int(sys.argv[3])
    #   height = int(sys.argv[4])

    dim0 = []
    dim1 = []
    splits = []

    for i in ['test', 'train']:

        # save_dir = sys.argv[2] + i + '/'
        save_dir = f'C:/Users/mazhar/check/{i}/'
        os.makedirs(save_dir, exist_ok=True)

        for root, dirs, files in tqdm(os.walk(f'{input_path}{i}')):
            for file in files:
                # set keep_ratio=True to have original aspect ratio
                xray = read_xray(os.path.join(root, file))
                img = resize(xray, width, height)
                img.save(os.path.join(save_dir, file.replace('dcm', 'png')))

                image_id.append(file.replace('.dcm', ''))
                dim0.append(xray.shape[0])
                dim1.append(xray.shape[1])
                splits.append(i)

    df = pd.DataFrame.from_dict({'image_id': image_id, 'dim0': dim0, 'dim1': dim1, 'split': splits})
    csv_file_save_path = 'C:/Users/mazhar/check/' + '/' + 'meta.csv'
    print(csv_file_save_path)
    df.to_csv(csv_file_save_path, index=False)


if __name__ == "__main__":
    main()
