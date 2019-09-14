import os

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
import pandas as pd

from sub_func.get_vgg16 import get_vgg16
from sub_func.layer import rpn_layer, classifier_layer
from sub_func.loss_func import rpn_loss_regr, rpn_loss_cls, class_loss_regr, class_loss_cls
from config import optimizer, optimizer_classifier, SAVE_RECORD_PATH


def build_model(config, cls_cnt):
    img_input = Input(shape=(None, None, 3))
    roi_input = Input(shape=(None, 4))

    num_anchors = len(config.anchor_box_scales) * len(config.anchor_box_ratios)  # 3 x 3 = 9
    # Define the base network (VGG here, can be Resnet50, Inception, etc)
    shared_layers = get_vgg16(img_input)

    # RPN Model
    rpn_out_class, rpn_out_regress = rpn_layer(shared_layers, num_anchors)
    model_rpn = Model(inputs=img_input, outputs=[rpn_out_class, rpn_out_regress])

    # Classifier Model
    classifier_out_class_softmax, classifier_out_bbox_linear_regression = classifier_layer(
        shared_layers,
        roi_input,
        config.num_rois,
        nb_classes=len(cls_cnt)
    )
    model_classifier = Model(
        inputs=[img_input, roi_input],
        outputs=[classifier_out_class_softmax, classifier_out_bbox_linear_regression]
    )

    # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
    model_all = Model(
        inputs=[img_input, roi_input],
        outputs=[rpn_out_class, rpn_out_regress, classifier_out_class_softmax, classifier_out_bbox_linear_regression]
    )

    # Load Weights From Saved Model or Base Network
    # 1. If Initial Train (There is no saved model [config.model_path]) ==> Load config.base_net_weights
    if not os.path.isfile(config.model_path):
        # If this is the begin of the training, load the pre-traind base network such as vgg-16
        try:
            print('\nThis is the first time of your training')
            print('Load weights from {}'.format(config.base_net_weights))
            model_rpn.load_weights(config.base_net_weights, by_name=True)
            model_classifier.load_weights(config.base_net_weights, by_name=True)
        except:
            print('Could not load pretrained model weights. Weights can be found in the keras application folder \
                https://github.com/fchollet/keras/tree/master/keras/applications')

        # Create the record.csv file to record losses, acc and mAP
        df_record = pd.DataFrame(
            columns=[
                'mean_overlapping_bboxes',
                'class_acc',
                'loss_rpn_cls',
                'loss_rpn_regr',
                'loss_class_cls',
                'loss_class_regr',
                'curr_loss',
                'elapsed_time',
                'mAP'
            ]
        )
    else:
        # 2. Continued Training (There is a saved model [config.model_path])
        print('\nContinue training based on previous trained model')
        print('Loading weights from {}'.format(config.model_path))
        model_rpn.load_weights(config.model_path, by_name=True)
        model_classifier.load_weights(config.model_path, by_name=True)

        df_record = pd.read_csv(SAVE_RECORD_PATH)
        print('Already train %dK batches' % (len(df_record)))

    # Model Compile
    model_rpn.compile(optimizer=optimizer, loss=[rpn_loss_cls(num_anchors), rpn_loss_regr(num_anchors)])
    model_classifier.compile(
        optimizer=optimizer_classifier,
        loss=[
            class_loss_cls,
            class_loss_regr(len(cls_cnt)-1),
        ],
        metrics={f'dense_class_{len(cls_cnt)}': 'accuracy'},
    )
    model_all.compile(optimizer='sgd', loss='mae')
    return model_rpn, model_classifier, model_all, df_record
