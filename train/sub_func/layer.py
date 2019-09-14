import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer, Dense, Conv2D, MaxPooling2D, TimeDistributed, Flatten, Dropout


_all__ = ['RoiPoolingConv', 'rpn_layer', 'classifier_layer']


class RoiPoolingConv(Layer):
    """ROI pooling layer for 2D inputs.
    See Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition,
    K. He, X. Zhang, S. Ren, J. Sun
    # Arguments
        pool_size: int
            Size of pooling region to use. pool_size = 7 will result in a 7x7 region.
        num_rois: number of regions of interest to be used
    # Input shape
        list of two 4D tensors [X_img, X_roi] with shape:
        X_img:
        `(1, rows, cols, channels)`
        X_roi:
        `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
    # Output shape
        3D tensor with shape:
        `(1, num_rois, channels, pool_size, pool_size)`
    """
    # Use Like Below (In the classifier_layer)
    # # ### input_shape = (num_rois, 7, 7, 512)
    # pooling_regions = 7
    # # ### out_roi_pool.shape = (1, num_rois, channels, pool_size, pool_size)
    # # ### num_rois (4) 7x7 roi pooling
    # out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    def __init__(self, pool_size, num_rois, **kwargs):
        self.pool_size = pool_size
        self.num_rois = num_rois
        super(RoiPoolingConv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.nb_channels = input_shape[0][3]

    def compute_output_shape(self, input_shape):
        return None, self.num_rois, self.pool_size, self.pool_size, self.nb_channels

    def call(self, x):
        assert (len(x) == 2)
        # x[0] is image with shape (rows, cols, channels)
        img = x[0]
        # x[1] is roi with shape (num_rois, 4) with ordering (x, y, w, h)
        rois = x[1]
        outputs = list()

        for roi_idx in range(self.num_rois):
            x = rois[0, roi_idx, 0]
            y = rois[0, roi_idx, 1]
            w = rois[0, roi_idx, 2]
            h = rois[0, roi_idx, 3]

            x = K.cast(x, dtype='int32')
            y = K.cast(y, dtype='int32')
            w = K.cast(w, dtype='int32')
            h = K.cast(h, dtype='int32')

            # Resized roi of the image to pooling size (7 x 7)
            rs = tf.image.resize_images(img[:, y:y + h, x:x + w, :], (self.pool_size, self.pool_size))
            outputs.append(rs)

        final_output = K.concatenate(outputs, axis=0)

        # Reshape to (1, num_rois, pool_size, pool_size, nb_channels)
        # Might be (1, 4, 7, 7, 3)
        final_output = K.reshape(final_output, (1, self.num_rois, self.pool_size, self.pool_size, self.nb_channels))

        # permute_dimensions is similar to transpose (np.transpose)
        final_output = K.permute_dimensions(final_output, (0, 1, 2, 3, 4))
        return final_output

    def get_config(self):
        config = {
            'pool_size': self.pool_size,
            'num_rois': self.num_rois,
        }
        base_config = super(RoiPoolingConv, self).get_config()
        return dict(
            list(base_config.items()) + list(config.items())
        )


def rpn_layer(base_layers, num_anchors):
    """Create a rpn layer
        Step1: Pass through the feature map from base layer to a 3x3 512 channels convolutional layer
                Keep the padding 'same' to preserve the feature map's size
        Step2: Pass the step1 to two (1,1) convolutional layer to replace the fully connected layer
                classification layer: num_anchors (9 in here) channels for 0, 1 sigmoid activation output
                regression layer: num_anchors*4 (36 in here) channels for computing the regression of bboxes with linear activation
    Args:
        base_layers: vgg in here
        num_anchors: 9 in here

    Returns:
        [x_class, x_regr, base_layers]
        x_class: classification for whether it's an object
        x_regr: bboxes regression
        base_layers: vgg in here
    """
    x = Conv2D(
        512,
        (3, 3),
        padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1'
    )(base_layers)
    # x.shape = (?, ?, ?, 512)
    rpn_out_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    rpn_out_regress = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)
    # rpn_out_class.shape = (?, ?, ?, 9)
    # rpn_out_regress.shape = (?, ?, ?, 36)
    return rpn_out_class, rpn_out_regress


def classifier_layer(base_layers, input_rois, num_rois, nb_classes=4):
    """Create a classifier layer

    Args:
        base_layers: vgg
        input_rois: `(1,num_rois,4)` list of rois, with ordering (x,y,w,h)
        num_rois: number of rois to be processed in one time (4 in here)

    Returns:
        list(out_class, out_regr)
        out_class: classifier layer output
        out_regr: regression layer output
    """

    # input_shape = (num_rois, 7, 7, 512)
    pooling_regions = 7
    # out_roi_pool.shape = (1, num_rois, channels, pool_size, pool_size)
    # num_rois (4) 7x7 roi pooling
    out_roi_pool = RoiPoolingConv(pooling_regions, num_rois)([base_layers, input_rois])

    # Flatten the convlutional layer and connected to 2 FC and 2 dropout
    out = TimeDistributed(Flatten(name='flatten'))(out_roi_pool)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc1'))(out)
    out = TimeDistributed(Dropout(0.5))(out)
    out = TimeDistributed(Dense(4096, activation='relu', name='fc2'))(out)
    out = TimeDistributed(Dropout(0.5))(out)

    # There are two output layer
    # out_class: softmax acivation function for classify the class name of the object
    # out_regr: linear activation function for bboxes coordinates regression
    classifier_out_class_softmax = TimeDistributed(Dense(
        units=nb_classes, activation='softmax', kernel_initializer='zero'),
        name='dense_class_{}'.format(nb_classes)
    )(out)
    # note: no regression target for bg class
    classifier_out_bbox_linear_regression = TimeDistributed(Dense(
        units=4 * (nb_classes - 1), activation='linear', kernel_initializer='zero'),
        name='dense_regress_{}'.format(nb_classes)
    )(out)
    return classifier_out_class_softmax, classifier_out_bbox_linear_regression
