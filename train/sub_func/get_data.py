import sys
import cv2


def get_data(fpath):
    """Parse the data from annotation file

    Args:
        fpath: annotation file path

    Returns:
        img_data_list: list(img_path, width, height, list(bboxes))
        cls_cnt: dict{key:class_name, value:count_num}
            e.g. {'Car': 30, 'Mobile phone': 73, 'Person': 122}
        cls_mapping: dict{key:class_name, value: idx}
            e.g. {'Car': 0, 'Mobile phone': 1, 'Person': 2}
    """
    img_info = dict()
    cls_cnt = dict()
    cls_mapping = dict()
    found_bg = False

    print('#### Start Get Data From Annotation File ####')
    with open(fpath, 'rt') as f:
        for idx, line in enumerate(f, 1):
            # --- Print Process ---
            sys.stdout.write('\r' + '* Current idx: ' + str(idx))
            # --- ----- ------- ---
            line_split = line.strip().split(',')
            img_path, img_mask_path, x1, y1, x2, y2, cls_name, predicted_iou = line_split

            # Class Count
            if cls_name not in cls_cnt:
                cls_cnt[cls_name] = 1
            else:
                cls_cnt[cls_name] += 1

            # Class Mapping
            if cls_name not in cls_mapping:
                if cls_name == 'bg' and found_bg == False:
                    print("Class Name ('bg') will be treated as a backgroud region")
                    found_bg = True
                cls_mapping[cls_name] = len(cls_mapping)  # 첫 Class 는 0, 이후 1부터 증가

            # Image File Info Dictionary
            if img_path not in img_info:
                img = cv2.imread(img_path)
                rows, cols = img.shape[:2]

                img_info[img_path] = dict(
                    filepath=img_path,
                    width=cols,
                    height=rows,
                    bboxes=list(),
                )

            # Append bbox data
            img_info[img_path]['bboxes'].append(
                {
                    'class': cls_name,
                    'x1': int(x1),
                    'x2': int(x2),
                    'y1': int(y1),
                    'y2': int(y2)
                }
            )

        # Convert img_info to List
        img_data_list = list(img_info.values())

        # Make sure the bg class is last in the list
        if found_bg and max(cls_mapping) != 'bg':
            key_to_switch = max(cls_mapping)  # Index 가 가장 큰 Class
            val_to_switch = cls_mapping['bg']  # 현재 bg 의 Index
            cls_mapping[key_to_switch], cls_mapping['bg'] = val_to_switch, len(cls_mapping) - 1

        # Make sure cls_cnt and cls_mapping have the bg class
        if not found_bg:
            cls_cnt['bg'] = 0
            cls_mapping['bg'] = len(cls_mapping)

    print('\n#### End of Get Data From Annotation File ####\n')
    return img_data_list, cls_cnt, cls_mapping
