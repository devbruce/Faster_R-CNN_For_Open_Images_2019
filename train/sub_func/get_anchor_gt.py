import numpy as np
import cv2
from .calc_rpn import calc_rpn
from .img_prep import get_new_img_size, augment


__all__ = ['get_anchor_gt']


def get_anchor_gt(img_data_list, config, resize_func, mode='train'):
    """ Yield the ground-truth anchors as Y (labels)

    Args:
        img_data_list: list(filepath, width, height, list(bboxes))
        config: Config instance
        resize_func: function to calculate final layer's feature map (of base model) size according to input image size
        mode: 'train' or 'test'; 'train' mode need augmentation

    Returns:
        img_data_prep: image data after resized and scaling (smallest size = 300px)
        Y: [y_rpn_cls, y_rpn_regr]
        img_data_aug: augmented image data (original image with augmentation)
        debug_img: show image for debug
        num_pos: show number of positive anchors for debug
    """
    while True:
        for img_data in img_data_list:
            try:
                if mode == 'train':
                    img_data_aug, img_data_prep = augment(img_data, config, augment=True)
                else:
                    img_data_aug, img_data_prep = augment(img_data, config, augment=False)

                width, height = img_data_aug['width'], img_data_aug['height']
                rows, cols = img_data_prep.shape[:2]

                assert cols == width
                assert rows == height

                # get image dimensions for resizing
                resized_width, resized_height = get_new_img_size(width, height, config.im_size)

                # resize the image so that smalles side is length = 300px
                img_data_prep = cv2.resize(img_data_prep, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
                debug_img = img_data_prep.copy()

                try:
                    y_rpn_cls, y_rpn_regr, num_pos = calc_rpn(
                        config,
                        img_data_aug,
                        width,height,
                        resized_width,
                        resized_height,
                        resize_func,
                    )
                except:
                    continue

                # Zero-center by mean pixel, and preprocess image
                img_data_prep = img_data_prep[:, :, (2, 1, 0)]  # BGR -> RGB
                img_data_prep = img_data_prep.astype(np.float32)
                img_data_prep[:, :, 0] -= config.img_channel_mean[0]
                img_data_prep[:, :, 1] -= config.img_channel_mean[1]
                img_data_prep[:, :, 2] -= config.img_channel_mean[2]
                img_data_prep /= config.img_scaling_factor

                img_data_prep = np.transpose(img_data_prep, (2, 0, 1))
                img_data_prep = np.expand_dims(img_data_prep, axis=0)

                y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= config.std_scaling

                img_data_prep = np.transpose(img_data_prep, (0, 2, 3, 1))
                y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
                y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))

                yield np.copy(img_data_prep), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug, debug_img, num_pos

            except Exception as e:
                print(e)
                continue
