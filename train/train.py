import warnings
import pickle
import random
import time

from tensorflow.keras.utils import Progbar
import numpy as np

from sub_func.get_data import get_data
from sub_func.get_anchor_gt import get_anchor_gt
from sub_func.img_prep import img_size_to_feature_map_size
from sub_func.rpn_to_roi import rpn_to_roi
from sub_func.calc_iou import calc_iou
from config import *
from build_model import build_model


warnings.filterwarnings('ignore')
random.seed(2019)
PROGRESS_VERBOSE = True

img_data_list, cls_cnt, cls_mapping = get_data(TRAIN_ANNOTATION_FILE_PATH)
C = Config()
C.cls_mapping = cls_mapping
# Shuffle the images with seed
random.shuffle(img_data_list)
# Get train data generator which generate X, Y, image_data
train_data_gen = get_anchor_gt(img_data_list, C, img_size_to_feature_map_size, mode='train')

if PROGRESS_VERBOSE:
    print(f'\nTraining images per class: {cls_cnt}')
    print(f'Number of Classes (Including bg) = {len(cls_cnt)}')
    print(f'\n--- Class Index Mapping ---\n{cls_mapping}')

# Save the configuration
with open(SAVE_CONFIG_PATH, 'wb') as config_file:
    pickle.dump(C, config_file)
    print(f'\nConfig has been written to\n==> ({SAVE_CONFIG_PATH}),'
          '\nand can be loaded when testing to ensure correct results\n')

# Build Model
model_rpn, model_classifier, model_all, df_record = build_model(config=C, cls_cnt=cls_cnt)

# ==== --- Training Process --- ==== #
# Training setting
total_epochs = len(df_record)
r_epochs = len(df_record)

epoch_length = 1000
num_epochs = 40
iter_num = 0

total_epochs += num_epochs

rpn_accuracy_rpn_monitor = list()
rpn_accuracy_for_epoch = list()
losses = np.zeros((epoch_length, 5))
best_loss = np.Inf if len(df_record) == 0 else np.min(df_record['curr_loss'])

# === Training ====
start_time = time.time()
for epoch_num in range(num_epochs):
    print('Epoch {}/{}'.format(r_epochs+1, total_epochs))
    progbar = Progbar(epoch_length)
    r_epochs += 1

    while True:
        try:
            if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor)) / len(rpn_accuracy_rpn_monitor)
                rpn_accuracy_rpn_monitor = list()
                # print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
                if mean_overlapping_bboxes == 0:
                    print(
                        '!! RPN is not producing bounding boxes that overlap the ground truth boxes.\n'
                        '\tCheck RPN settings or keep training. !!'
                    )

            # Generate X (x_img) and label Y ([y_rpn_cls, y_rpn_regr])
            # yield np.copy(img_aug), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)], img_data_aug, debug_img, num_pos
            X_train_img, X_train_img_data, Y_train, debug_img, debug_num_pos = next(train_data_gen)

            # Train rpn model and get loss value [_, loss_rpn_cls, loss_rpn_regr]
            loss_rpn = model_rpn.train_on_batch(X_train_img, Y_train)

            # Get predicted rpn from rpn model [rpn_cls, rpn_regr]
            Y_hat_rpn = model_rpn.predict_on_batch(x=X_train_img)
            Y_hat_rpn_cls = Y_hat_rpn[0]
            Y_hat_rpn_regr = Y_hat_rpn[1]
            # Y_hat_rpn_cls.shape: (1, 18, 28, 9)
            # Y_hat_rpn_regr.shape: (1, 18, 28, 36)

            # non_max_sup_bboxes: bboxes (shape=(300,4))
            # Convert rpn layer to roi bboxes
            non_max_sup_bboxes = rpn_to_roi(
                rpn_layer=Y_hat_rpn_cls,
                regr_layer=Y_hat_rpn_regr,
                config=C,
                use_regr=True,
                overlap_thresh=0.7,
                max_boxes=300,
            )
            # ===================================================================
            # rpn_layer: output layer for rpn classification
            #     shape (1, feature_map.height, feature_map.width, num_anchors)
            #     Might be (1, 18, 25, 18) if resized image is 400 width and 300
            # regr_layer: output layer for rpn regression
            #     shape (1, feature_map.height, feature_map.width, num_anchors)
            #     Might be (1, 18, 25, 72) if resized image is 400 width and 300
            # ===================================================================

            # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
            # X_train_roi: bboxes that iou > C.classifier_min_overlap for all gt bboxes in 300 non_max_suppression bboxes
            # Y_train_cls_num: one hot code for bboxes from above => x_roi (X)
            # Y_train_label_and_gt: corresponding labels and corresponding gt bboxes
            # calc_iou return np.expand_dims(X_roi, axis=0), np.expand_dims(Y_cls_num, axis=0), np.expand_dims(Y_label_and_gt, axis=0), IoUs
            X_train_roi, Y_train_cls_num, Y_train_label_and_gt, IouS = calc_iou(non_max_sup_bboxes, X_train_img_data, C, cls_mapping)

            # If X_train_roi is None means there are no matching bboxes
            if X_train_roi is None:
                rpn_accuracy_rpn_monitor.append(0)
                rpn_accuracy_for_epoch.append(0)
                continue

            # Find out the positive anchors and negative anchors
            neg_samples = np.where(Y_train_cls_num[0, :, -1] == 1)
            pos_samples = np.where(Y_train_cls_num[0, :, -1] == 0)

            neg_samples = neg_samples[0] if len(neg_samples) > 0 else list()
            pos_samples = pos_samples[0] if len(pos_samples) > 0 else list()

            rpn_accuracy_rpn_monitor.append(len(pos_samples))
            rpn_accuracy_for_epoch.append((len(pos_samples)))

            if C.num_rois > 1:
                # If number of positive anchors is larger than 4 // 2 = 2, randomly choose 2 pos samples
                if len(pos_samples) < C.num_rois // 2:
                    selected_pos_samples = pos_samples.tolist()
                else:
                    selected_pos_samples = np.random.choice(pos_samples, C.num_rois // 2, replace=False).tolist()

                # Randomly choose (num_rois - num_pos) neg samples
                try:
                    selected_neg_samples = np.random.choice(
                        neg_samples,
                        C.num_rois - len(selected_pos_samples),
                        replace=False
                    ).tolist()
                except:
                    selected_neg_samples = np.random.choice(
                        neg_samples,
                        C.num_rois - len(selected_pos_samples),
                        replace=True
                    ).tolist()
                # Save all the pos and neg samples in selected_samples
                selected_samples = selected_pos_samples + selected_neg_samples
            else:
                # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                selected_pos_samples = pos_samples.tolist()
                selected_neg_samples = neg_samples.tolist()
                if np.random.randint(0, 2):
                    selected_samples = random.choice(neg_samples)
                else:
                    selected_samples = random.choice(pos_samples)

            # training_data: [X, X_train_roi[:, selected_samples, :]]
            # labels: [Y_train_cls_num[:, selected_samples, :], Y_train_label_and_gt[:, selected_samples, :]]
            #  X                     => X_train_img_data resized image
            #  X_train_roi[:, selected_samples, :] => num_rois (4 in here) bboxes which contains selected neg and pos
            #  Y_train_cls_num[:, selected_samples, :] => one hot encode for num_rois bboxes which contains selected neg and pos
            #  Y_train_label_and_gt[:, selected_samples, :] => labels and gt bboxes for num_rois bboxes which contains selected neg and pos
            loss_class = model_classifier.train_on_batch(
                x=[X_train_img, X_train_roi[:, selected_samples, :]],  # training data
                y=[Y_train_cls_num[:, selected_samples, :], Y_train_label_and_gt[:, selected_samples, :]],  # target data
            )

            losses[iter_num, 0] = loss_rpn[1]  # loss_rpn_cls
            losses[iter_num, 1] = loss_rpn[2]  # loss_rpn_regr

            losses[iter_num, 2] = loss_class[1]  # loss_class_cls
            losses[iter_num, 3] = loss_class[2]  # loss_class_regr
            losses[iter_num, 4] = loss_class[3]  # class_acc

            iter_num += 1

            progbar.update(iter_num,
                           [
                               ('rpn_cls', np.mean(losses[:iter_num, 0])),
                               ('rpn_regr', np.mean(losses[:iter_num, 1])),
                               ('final_cls', np.mean(losses[:iter_num, 2])),
                               ('final_regr', np.mean(losses[:iter_num, 3])),
                           ])

            if iter_num == epoch_length:
                loss_rpn_cls = np.mean(losses[:, 0])
                loss_rpn_regr = np.mean(losses[:, 1])
                loss_class_cls = np.mean(losses[:, 2])
                loss_class_regr = np.mean(losses[:, 3])
                class_acc = np.mean(losses[:, 4])

                mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                rpn_accuracy_for_epoch = list()

                if C.verbose:
                    print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(
                        mean_overlapping_bboxes))
                    print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                    print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                    print('Loss RPN regression: {}'.format(loss_rpn_regr))
                    print('Loss Detector classifier: {}'.format(loss_class_cls))
                    print('Loss Detector regression: {}'.format(loss_class_regr))
                    print('Total loss: {}'.format(loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr))
                    print('Elapsed time: {}'.format(time.time() - start_time))
                    elapsed_time = (time.time() - start_time) / 60

                curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                iter_num = 0
                start_time = time.time()

                if curr_loss < best_loss:
                    if C.verbose:
                        print('Total loss decreased from {} to {}, saving weights'.format(best_loss, curr_loss))
                    best_loss = curr_loss
                    model_all.save_weights(C.model_path)

                new_row = {
                    'mean_overlapping_bboxes': round(mean_overlapping_bboxes, 3),
                    'class_acc': round(class_acc, 3),
                    'loss_rpn_cls': round(loss_rpn_cls, 3),
                    'loss_rpn_regr': round(loss_rpn_regr, 3),
                    'loss_class_cls': round(loss_class_cls, 3),
                    'loss_class_regr': round(loss_class_regr, 3),
                    'curr_loss': round(curr_loss, 3),
                    'elapsed_time': round(elapsed_time, 3),
                    'mAP': 0
                }

                df_record = df_record.append(new_row, ignore_index=True)
                df_record.to_csv(SAVE_RECORD_PATH, index=0)
                break

        except Exception as e:
            print('Exception: {}'.format(e))
            continue

print('=== Training complete! ===')
