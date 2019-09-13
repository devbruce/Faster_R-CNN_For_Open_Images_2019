import numpy as np
from sub_func.img_prep import get_new_img_size
from sub_func.get_iou import iou
import copy


def calc_iou(non_max_sup_bboxes, img_data, config, cls_mapping):
    """Converts from (x1,y1,x2,y2) to (x,y,w,h) format

    Args:
        non_max_sup_bboxes: return value of `rpn_to_roi` function
    """
    gt_bboxes = img_data['bboxes']
    nb_gt_bboxes = len(gt_bboxes)
    width, height = img_data['width'], img_data['height']
    # get image dimensions for resizing
    resized_width, resized_height = get_new_img_size(width, height, config.im_size)

    gta = np.zeros((nb_gt_bboxes, 4))

    for bbox_num, bbox in enumerate(gt_bboxes):
        # get the GT box coordinates, and resize to account for image resizing
        # gta[bbox_num, 0] = (40 * (600 / 800)) / 16 = int(round(1.875)) = 2 (x in feature map)
        gta[bbox_num, 0] = int(round(bbox['x1'] * (resized_width / float(width)) / config.rpn_stride))
        gta[bbox_num, 1] = int(round(bbox['x2'] * (resized_width / float(width)) / config.rpn_stride))
        gta[bbox_num, 2] = int(round(bbox['y1'] * (resized_height / float(height)) / config.rpn_stride))
        gta[bbox_num, 3] = int(round(bbox['y2'] * (resized_height / float(height)) / config.rpn_stride))

    x_roi = list()
    y_class_num = list()  # One Hot Encoding of Class Number
    y_class_regr_coords = list()
    y_class_regr_label = list()
    IoUs = list()  # for debugging only

    # non_max_sup_bboxes.shape[0]: number of bboxes (=300 from non_max_suppression)
    for ix in range(non_max_sup_bboxes.shape[0]):
        (x1, y1, x2, y2) = non_max_sup_bboxes[ix, :]
        x1 = int(round(x1))
        y1 = int(round(y1))
        x2 = int(round(x2))
        y2 = int(round(y2))

        best_iou = 0.0
        best_bbox = -1
        # Iterate through all the ground-truth bboxes to calculate the iou
        for bbox_num in range(nb_gt_bboxes):
            curr_iou = iou([gta[bbox_num, 0], gta[bbox_num, 2], gta[bbox_num, 1], gta[bbox_num, 3]], [x1, y1, x2, y2])

            # Find out the corresponding ground-truth bbox_num with larget iou
            if curr_iou > best_iou:
                best_iou = curr_iou
                best_bbox = bbox_num

        if best_iou < config.classifier_min_overlap:
            continue
        else:
            w = x2 - x1
            h = y2 - y1
            x_roi.append([x1, y1, w, h])
            IoUs.append(best_iou)

            if config.classifier_min_overlap <= best_iou < config.classifier_max_overlap:  # 0.3 <= best_iou < 0.7
                # hard negative example
                cls_name = 'bg'
            elif best_iou >= config.classifier_max_overlap:  # best_iou >= 0.7
                cls_name = gt_bboxes[best_bbox]['class']
                cxg = (gta[best_bbox, 0] + gta[best_bbox, 1]) / 2.0
                cyg = (gta[best_bbox, 2] + gta[best_bbox, 3]) / 2.0

                cx = x1 + w / 2.0
                cy = y1 + h / 2.0

                tx = (cxg - cx) / float(w)
                ty = (cyg - cy) / float(h)
                tw = np.log((gta[best_bbox, 1] - gta[best_bbox, 0]) / float(w))
                th = np.log((gta[best_bbox, 3] - gta[best_bbox, 2]) / float(h))
            else:
                print('roi = {}'.format(best_iou))
                raise RuntimeError

        nb_cls = len(cls_mapping)

        # === class_num --> One Hot Encoding Process ===
        class_num = cls_mapping[cls_name]
        class_label = [0] * nb_cls
        class_label[class_num] = 1
        # Append One Hot Encoded Class Number to y_class_num List
        y_class_num.append(copy.deepcopy(class_label))
        # === ====================================== ===

        coords = [0] * 4 * (nb_cls - 1)
        labels = [0] * 4 * (nb_cls - 1)
        if cls_name != 'bg':
            label_pos = 4 * class_num
            sx, sy, sw, sh = config.classifier_regr_std  # [8.0, 8.0, 4.0, 4.0]
            coords[label_pos:4+label_pos] = [tx * sx, ty * sy, tw * sw, th * sh]
            labels[label_pos:4+label_pos] = [1, 1, 1, 1]
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))
        else:
            y_class_regr_coords.append(copy.deepcopy(coords))
            y_class_regr_label.append(copy.deepcopy(labels))

    if len(x_roi) == 0:
        return None, None, None, None

    # bboxes that iou > config.classifier_min_overlap for all gt bboxes in 300 non_max_suppression bboxes
    X = np.array(x_roi)
    # one hot code for bboxes from above => x_roi (X)
    Y1 = np.array(y_class_num)
    # corresponding labels and corresponding gt bboxes
    Y2 = np.concatenate([np.array(y_class_regr_label), np.array(y_class_regr_coords)], axis=1)

    return np.expand_dims(X, axis=0), np.expand_dims(Y1, axis=0), np.expand_dims(Y2, axis=0), IoUs
