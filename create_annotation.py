import os
import sys
import pandas as pd
import cv2


print('\n=== Create Annotation File ===\n')
Q_ANS = input('Enter "Full" to make text annotation file with full data.\n'
      'If you want to use lighrweight data for testing, enter any value\n')
USE_FULL_DATA = True if Q_ANS == 'Full' else False
print('** Use Full Data **\n') if USE_FULL_DATA else print('** Use Lightweight Data **\n')

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, 'data')
CHALLENGE_2019_DIR = os.path.join(DATA_DIR, 'challenge-2019')
TRAIN_IMG_DIR = os.path.join(DATA_DIR, 'train_img')
TRAIN_MASK_IMG_DIR = os.path.join(DATA_DIR, 'train_mask_img')

CLASS_DESCRIPTION_FILE_PATH = os.path.join(CHALLENGE_2019_DIR, 'challenge-2019-classes-description-segmentable.csv')
if USE_FULL_DATA:
    MASK_ANNOTATION_FILE_PATH = os.path.join(CHALLENGE_2019_DIR, 'challenge-2019-train-segmentation-masks.csv')
else:
    MASK_ANNOTATION_FILE_PATH = os.path.join(DATA_DIR, 'train-segmentation-masks_light.csv')
TRAIN_ANNOTATION_FILE_PATH = os.path.join(DATA_DIR, 'train_annotation.txt')

df_class_descriptions = pd.read_csv(CLASS_DESCRIPTION_FILE_PATH, names=['LabelName', 'ClassName'])
label_mapping_to_class_name = dict(zip(df_class_descriptions['LabelName'], df_class_descriptions['ClassName']))
df_mask_annotation = pd.read_csv(MASK_ANNOTATION_FILE_PATH)

# Write Train Annotation File
f = open(TRAIN_ANNOTATION_FILE_PATH, 'wt')

nb_object = len(df_mask_annotation)
for idx, row in df_mask_annotation.iterrows():
    img_file_path = os.path.join(TRAIN_IMG_DIR, row['ImageID'] + '.jpg')
    mask_img_file_path = os.path.join(TRAIN_MASK_IMG_DIR, row['MaskPath'])

    # Read Image For Getting Size (Height, Width)
    img = cv2.imread(img_file_path)
    height, width = img.shape[:2]

    # Computed BBox Coordinate
    x1 = int(row['BoxXMin'] * width)
    x2 = int(row['BoxXMax'] * width)
    y1 = int(row['BoxYMin'] * height)
    y2 = int(row['BoxYMax'] * height)

    label_name = row['LabelName']
    class_name = label_mapping_to_class_name[label_name]
    predicted_iou = row['PredictedIoU']
    clicks = row['Clicks']

    cur_row = img_file_path + ',' + \
              mask_img_file_path + ',' + \
              str(x1) + ',' + \
              str(y1) + ',' + \
              str(x2) + ',' + \
              str(y2) + ',' + \
              class_name + ',' + \
              str(predicted_iou) + ',' + \
              clicks + '\n'
    f.write(cur_row)
    sys.stdout.write(f'\r - Get Train Annotation File Progress: {str(idx+1)} / {nb_object}')

f.close()
