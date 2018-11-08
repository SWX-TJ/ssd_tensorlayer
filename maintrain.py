import numpy as np
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.files import maybe_download_and_extract
from tensorlayer.files import assign_params
import random
from l2_normalization import L2_normlization_Layer
import numpy as np
import math
import os
from imutils.object_detection import non_max_suppression

# Super Param
learning_rate = 0.00001
batch_size = 16
train_epoch = 10000
feat_layers = ['block4', 'block7', 'block8', 'block9', 'block10', 'block11']
feature_maps_shape = [(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)]
# 最小default box面积比例
min_box_scale = 0.1
# 最大default box面积比例
max_box_scale = 0.9
default_box_size = [4, 6, 6, 6, 4, 4]
box_aspect_ratio = [
    [0.5, 1.0, 2.0],
    [0.5, 1.0, 2.0, 3.0, 1 / 3.0],
    [0.5, 1.0, 2.0, 3.0, 1 / 3.0],
    [0.5, 1.0, 2.0, 3.0, 1 / 3.0],
    [0.5, 1.0, 2.0],
    [0.5, 1.0, 2.0]
]
# pred_class
all_num_classes = 21
# feature is need NormLizations
normalizations = [10, -1, -1, -1, -1, -1, -1]
# 每个特征层的面积比例
# 论文中的s_k=s_min+(s_max-s_min)*(k-1)/(m-1)
default_box_scale = np.array((0.05, 0.1, 0.3, 0.5, 0.7, 0.9))
all_default_boxs_len = 0
all_default_boxs = []


# 产生default box
def generate_all_default_boxs():
    all_default_boxes = []
    for index, map_shape in zip(range(len(feature_maps_shape)), feature_maps_shape):
        width = int(map_shape[0])
        height = int(map_shape[1])
        cell_scale = default_box_scale[index]
        for x in range(width):
            for y in range(height):
                for ratio in box_aspect_ratio[index]:
                    center_x = (x / float(width)) + (0.5 / float(width))
                    center_y = (y / float(height)) + (0.5 / float(height))
                    box_width = cell_scale * np.sqrt(ratio)
                    box_height = cell_scale / np.sqrt(ratio)
                    all_default_boxes.append([center_x, center_y, box_width, box_height])
                if index == len(feature_maps_shape) - 1:
                    all_default_boxes.append(
                        [(x / float(width)) + (0.5 / float(width)), (y / float(height)) + (0.5 / float(height)),
                         np.sqrt(0.9 * 1.05), np.sqrt(0.9 * 1.05)])
                else:
                    all_default_boxes.append(
                        [(x / float(width)) + (0.5 / float(width)), (y / float(height)) + (0.5 / float(height)),
                         np.sqrt(default_box_scale[index + 1] * cell_scale),
                         np.sqrt(default_box_scale[index + 1] * cell_scale)])
    all_default_boxes = np.array(all_default_boxes)
    return all_default_boxes


# Match Stratgy
def generate_groundtruth_data(input_actual_data):
    # 生成空数组，用于保存groundtruth
    input_actual_data_len = len(input_actual_data)  # 16
    t_class = np.zeros((input_actual_data_len, all_default_boxs_len))  # (16,8732)
    t_location = np.zeros((input_actual_data_len, all_default_boxs_len, 4))  # (16,8732,4)
    t_positives_jacc = np.zeros((input_actual_data_len, all_default_boxs_len))  # (16,8732)
    t_positives = np.zeros((input_actual_data_len, all_default_boxs_len))  # (16,8732)
    t_negatives = np.zeros((input_actual_data_len, all_default_boxs_len))  # (16,8732)
    background_jacc = max(0, 0.2)
    # 1.gt_bbox--->best jac val default_bbox
    for img_index in range(input_actual_data_len):
        for per_gt_bbox in input_actual_data[img_index]:
            gt_class = per_gt_bbox[-1:][0]
            gt_bbox = per_gt_bbox[:-1]
            best_jacc_index = -1
            default_jacc = 0.5
            for default_bbox_index in range(all_default_boxs_len):
                jacc, jacc_encode_loc = jaccard(gt_bbox, all_default_boxs[default_bbox_index])
                if jacc >= default_jacc:
                    default_jacc = jacc
                    best_jacc_index = default_bbox_index
            if best_jacc_index != -1:
                best_jacc, best_jacc_encode_loc = jaccard(gt_bbox, all_default_boxs[best_jacc_index])
                t_class[img_index][best_jacc_index] = gt_class+1
                t_location[img_index][best_jacc_index] = best_jacc_encode_loc
                t_positives_jacc[img_index][best_jacc_index] = best_jacc
                t_positives[img_index][best_jacc_index] = 1
                t_negatives[img_index][best_jacc_index] = 0
    # 2.left default_bbox--->IOU>threshold gt_bbox
    for img_index in range(input_actual_data_len):
        for left_bbox_index in range(len(t_positives[img_index])):
            if (t_positives[img_index][left_bbox_index] == 0):
                best_jacc = 0.5
                best_index = 0
                flag = 0
                for per_gt_bbox_index in range(len(input_actual_data[img_index])):
                    gt_bbox = input_actual_data[img_index][per_gt_bbox_index][:-1]
                    jacc, jacc_encode_loc = jaccard(gt_bbox, all_default_boxs[left_bbox_index])
                    if jacc > best_jacc:
                        best_jacc = jacc
                        best_index = per_gt_bbox_index
                        flag = 1
                if flag == 1:
                    flag = 0
                    best_jacca, best_jacc_encode_loca = jaccard(input_actual_data[img_index][best_index][:-1],
                                                                all_default_boxs[left_bbox_index])
                    #print("best_jacc_encode_loca",best_jacc_encode_loca)
                    t_class[img_index][left_bbox_index] = input_actual_data[img_index][best_index][-1:][0]+1
                    t_location[img_index][left_bbox_index] = best_jacc_encode_loca
                    t_positives_jacc[img_index][left_bbox_index] = best_jacca
                    t_positives[img_index][left_bbox_index] = 1
                    t_negatives[img_index][left_bbox_index] = 0
    # 3.Hard sample select
    for img_index in range(input_actual_data_len):
        gt_neg_end_count = int(np.sum(t_positives[img_index]) * 3)
        if (gt_neg_end_count + np.sum(t_positives[img_index])) > all_default_boxs_len:
            gt_neg_end_count = all_default_boxs_len - np.sum(t_positives[img_index])
        gt_neg_index = np.random.randint(low=0, high=all_default_boxs_len, size=int(gt_neg_end_count))
        for r_index in gt_neg_index:
            if t_positives_jacc[img_index][r_index] < background_jacc and t_positives[img_index][r_index] != 1:
                t_class[img_index][r_index] = 0  # 背景类
                t_positives[img_index][r_index] = 0
                t_negatives[img_index][r_index] = 1
    return t_class, t_location, t_positives, t_negatives


def jaccard(rect1, rect2):
    x_overlap = max(0, (min(rect1[0] + (rect1[2] / 2), rect2[0] + (rect2[2] / 2)) - max(rect1[0] - (rect1[2] / 2),
                                                                                        rect2[0] - (rect2[2] / 2))))
    y_overlap = max(0, (min(rect1[1] + (rect1[3] / 2), rect2[1] + (rect2[3] / 2)) - max(rect1[1] - (rect1[3] / 2),
                                                                                        rect2[1] - (rect2[3] / 2))))
    intersection = x_overlap * y_overlap
    # 删除超出图像大小的部分
    # rect1_width_sub = 0
    # rect1_height_sub = 0
    # rect2_width_sub = 0
    # rect2_height_sub = 0
    # if (rect1[0] - rect1[2] / 2) < 0: rect1_width_sub += 0 - (rect1[0] - rect1[2] / 2)
    # if (rect1[0] + rect1[2] / 2) > 1: rect1_width_sub += (rect1[0] + rect1[2] / 2) - 1
    # if (rect1[1] - rect1[3] / 2) < 0: rect1_height_sub += 0 - (rect1[1] - rect1[3] / 2)
    # if (rect1[1] + rect1[3] / 2) > 1: rect1_height_sub += (rect1[1] + rect1[3] / 2) - 1
    # if (rect2[0] - rect2[2] / 2) < 0: rect2_width_sub += 0 - (rect2[0] - rect2[2] / 2)
    # if (rect2[0] + rect2[2] / 2) > 1: rect2_width_sub += (rect2[0] + rect2[2] / 2) - 1
    # if (rect2[1] - rect2[3] / 2) < 0: rect2_height_sub += 0 - (rect2[1] - rect2[3] / 2)
    # if (rect2[1] + rect2[3] / 2) > 1: rect2_height_sub += (rect2[1] + rect2[3] / 2) - 1
    # area_box_a = (rect1[2] - rect1_width_sub) * (rect1[3] - rect1_height_sub)
    # area_box_b = (rect2[2] - rect2_width_sub) * (rect2[3] - rect2_height_sub)
    area_box_a = rect1[2] * rect1[3]
    area_box_b = rect2[2] * rect2[3]
    union = area_box_a + area_box_b - intersection
    if intersection > 0 and union > 0:
        return intersection / union, [(rect1[0] - (rect2[0])) / rect2[2], (rect1[1] - (rect2[1])) / rect2[3],
                                      math.log(rect1[2] / rect2[2]), math.log(rect1[3] / rect2[3])]

    else:
        return 0, [0.00001, 0.00001, 0.00001, 0.00001]


# Load Data(img,cls,loc)
im_size = [300, 300]
imgs_file_list, _, _, _, classes, _, _, _, objs_info_list, _ = tl.files.load_voc_dataset(dataset="2012")
laod_imgs = np.array(imgs_file_list)
ann_list = []
n_data = len(imgs_file_list)
print(n_data)
for info in objs_info_list:
    ann = tl.prepro.parse_darknet_ann_str_to_list(info)
    c, b = tl.prepro.parse_darknet_ann_list_to_cls_box(ann)
    ann_list.append([c, b])
load_ann = np.array(ann_list)


def getTrainData(batch_load_imgs, batch_load_ann):
    # idexs = tl.utils.get_random_int(min_v=0, max_v=n_data-1, number=batch_size)
    # b_im_path = [imgs_file_list[i] for i in batch_load_imgs]
    b_images = tl.prepro.threading_data(batch_load_imgs, fn=tl.vis.read_image)
    b_ann = batch_load_ann  # [ann_list[i] for i in batch_load_imgs]
    # print("b_ann",b_ann[0])
    def _data_pre_aug_fn(data):
        im, ann = data
        clas, coords = ann
        ## 随机改变图片亮度、对比度和饱和度
        im = tl.prepro.illumination(im, gamma=(0.5, 1.5),
                                    contrast=(0.5, 1.5), saturation=(0.5, 1.5), is_random=True)
        ## 随机左右翻转
        im, coords = tl.prepro.obj_box_horizontal_flip(im, coords,
                                                       is_rescale=True, is_center=True, is_random=True)
        ## 随机调整大小并裁剪出指定大小的图片，这同时达到了随机缩放的效果
        tmp0 = random.randint(1, int(im_size[0] * 0.3))
        tmp1 = random.randint(1, int(im_size[1] * 0.3))
        im, coords = tl.prepro.obj_box_imresize(im, coords,
                                                [im_size[0] + tmp0, im_size[1] + tmp1], is_rescale=True,
                                                interp='bicubic')
        im, clas, coords = tl.prepro.obj_box_crop(im, clas, coords,
                                                  wrg=im_size[1], hrg=im_size[0], is_rescale=True,
                                                  is_center=True, is_random=True)
        # im = tl.prepro.samplewise_norm(im, samplewise_center=True, samplewise_std_normalization=True)
        ## 把数值范围从 [0, 255] 转到 [-1, 1] (可选)
        # im = tl.prepro.pixel_value_scale(im, 0.1, [0, 255], is_random=True)
        # im = np.clip(im,0.0,1.0)
        im = im / 127.5 - 1.0
        return im, [clas, coords]

    data = tl.prepro.threading_data([_ for _ in zip(b_images, b_ann)],
                                    _data_pre_aug_fn)
    preproc_img = [d[0] for d in data]
    preproc_ann = [d[1] for d in data]
    preproc_label, preproc_bboxs = tl.prepro.parse_darknet_ann_list_to_cls_box(preproc_ann)
    actual_data = []
    for img_index in range(len(preproc_label)):
        aitems = []
        for i in range(len(preproc_label[img_index])):
            aitem = [preproc_bboxs[img_index][0][i][0], preproc_bboxs[img_index][0][i][1],
                     preproc_bboxs[img_index][0][i][2], preproc_bboxs[img_index][0][i][3], preproc_label[img_index][i]]
            aitems.append(aitem)
        actual_data.append(aitems)
    return preproc_img, actual_data


# Define Network
def ssd_multibox_layer(prev_Layer, num_classes, per_feat_anchors_num, normalizations):
    if normalizations > 0:
        network = L2_normlization_Layer(prev_Layer)
    else:
        network = prev_Layer
    Input_Channel = network.outputs.get_shape().as_list()[3]
    num_anchors = per_feat_anchors_num
    num_loc_pred = num_anchors * 4
    loc_pred = tl.layers.Conv2dLayer(network, act=tf.identity, shape=[3, 3, Input_Channel, num_loc_pred],
                                     strides=[1, 1, 1, 1],
                                     padding='SAME',
                                     W_init=tf.glorot_uniform_initializer(),
                                     name='conv_loc')
    rloc_pred = tl.layers.ReshapeLayer(loc_pred, [-1, loc_pred.outputs.get_shape().as_list()[1] *loc_pred.outputs.get_shape().as_list()[2] * num_anchors, 4],
                                      name='loc_pred_reshape')
    num_cls_pred = num_anchors * num_classes
    cls_pred = tl.layers.Conv2dLayer(network, act=tf.identity, shape=[3, 3, Input_Channel, num_cls_pred],
                                     strides=[1, 1, 1, 1],
                                     padding='SAME',
                                     W_init=tf.glorot_uniform_initializer(),
                                     name='conv_cls')
    rcls_pred = tl.layers.ReshapeLayer(cls_pred,[-1, cls_pred.outputs.get_shape().as_list()[1]*cls_pred.outputs.get_shape().as_list()[2] * num_anchors, num_classes],name='cls_pred_reshape')
    return rcls_pred, rloc_pred


def ssd_net(net_in, is_training=True, reuse=False):
    end_points = {}
    with tf.variable_scope("ssd_300_net", reuse=reuse):
        InputLayer = tl.layers.InputLayer(net_in, name='InputLayer')
        # vgg16_base
        # conv1
        network = tl.layers.Conv2dLayer(InputLayer, act=tf.nn.relu, shape=[3, 3, 3, 64], strides=[1, 1, 1, 1],
                                        padding='SAME', name='conv1_1')
        network = tl.layers.Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 64, 64], strides=[1, 1, 1, 1],
                                        padding='SAME', name='conv1_2')
        end_points['block1'] = network
        network = tl.layers.PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                      pool=tf.nn.max_pool, name='pool1')  # (-1,150,150,64)
        # conv2
        network = tl.layers.Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 64, 128], strides=[1, 1, 1, 1],
                                        padding='SAME', name='conv2_1')
        network = tl.layers.Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 128, 128], strides=[1, 1, 1, 1],
                                        padding='SAME', name='conv2_2')
        end_points['block2'] = network
        network = tl.layers.PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                      pool=tf.nn.max_pool, name='pool2')  # (-1,75,75,128)
        # conv3
        network = tl.layers.Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 128, 256], strides=[1, 1, 1, 1],
                                        padding='SAME', name='conv3_1')
        network = tl.layers.Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 256, 256], strides=[1, 1, 1, 1],
                                        padding='SAME', name='conv3_2')
        network = tl.layers.Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 256, 256], strides=[1, 1, 1, 1],
                                        padding='SAME', name='conv3_3')
        end_points['block3'] = network
        network = tl.layers.PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                      pool=tf.nn.max_pool, name='pool3')  # (-1,38,38,256)
        # conv4
        network = tl.layers.Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 256, 512], strides=[1, 1, 1, 1],
                                        padding='SAME', name='conv4_1')
        network = tl.layers.Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 512, 512], strides=[1, 1, 1, 1],
                                        padding='SAME', name='conv4_2')
        network = tl.layers.Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 512, 512], strides=[1, 1, 1, 1],
                                        padding='SAME', name='conv4_3')
        end_points['block4'] = network
        network = tl.layers.PoolLayer(network, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME',
                                      pool=tf.nn.max_pool, name='pool4')  # (-1,38,38,512)
        # conv5
        network = tl.layers.Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 512, 512], strides=[1, 1, 1, 1],
                                        padding='SAME', name='conv5_1')
        network = tl.layers.Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 512, 512], strides=[1, 1, 1, 1],
                                        padding='SAME', name='conv5_2')
        network = tl.layers.Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 512, 512], strides=[1, 1, 1, 1],
                                        padding='SAME', name='conv5_3')
        end_points['block5'] = network
        network = tl.layers.PoolLayer(network, ksize=[1, 3, 3, 1], strides=[1, 1, 1, 1], padding='SAME',
                                      pool=tf.nn.max_pool, name='pool5')  # (-1.38.38.512)
        # conv6
        network = tl.layers.AtrousConv2dLayer(network, 1024, [3, 3], rate=6, act=tf.nn.relu, name='conv6')
        #network = tl.layers.GroupNormLayer(network, groups=16, epsilon=1e-06, act=tf.nn.relu, data_format='channels_last',
                             #             name='groupnorm')
        end_points['block6'] = network
        # conv7
        network = tl.layers.Conv2dLayer(network, act=tf.nn.relu, shape=[1, 1, 1024, 1024], strides=[1, 1, 1, 1],
                                        padding='SAME', W_init=tf.glorot_uniform_initializer(),
                                        name='conv7')  # (-1,19,19,1024)
        end_points['block7'] = network
        # conv8
        end_point = 'block8'
        with tf.variable_scope(end_point):
            network = tl.layers.Conv2dLayer(network, act=tf.nn.relu, shape=[1, 1, 1024, 256], strides=[1, 1, 1, 1],
                                            padding='SAME', W_init=tf.glorot_uniform_initializer(), name='conv8_1')
            network = tl.layers.ZeroPad2d(network, padding=(1, 1), name='pad2d_1')
            network = tl.layers.Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 256, 512], strides=[1, 2, 2, 1],
                                            padding='VALID', W_init=tf.glorot_uniform_initializer(),
                                            name='conv8_2')  # (-1,10,10,512)
            # network  = tl.layers.BatchNormLayer(network,is_train = is_training, act =tf.nn.relu,name='block8_batchnorm_layer')
            end_points[end_point] = network
        # conv9
        end_point = 'block9'
        with tf.variable_scope(end_point):
            network = tl.layers.Conv2dLayer(network, act=tf.nn.relu, shape=[1, 1, 512, 128], strides=[1, 1, 1, 1],
                                            padding='SAME', W_init=tf.glorot_uniform_initializer(), name='conv9_1')
            network = tl.layers.ZeroPad2d(network, padding=(1, 1), name='pad2d_2')
            network = tl.layers.Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 128, 256], strides=[1, 2, 2, 1],
                                            padding='VALID', W_init=tf.glorot_uniform_initializer(),
                                            name='conv9_2')  # (-1,5,5,256)
            # network  = tl.layers.BatchNormLayer(network,is_train = is_training, act =tf.nn.relu,name='block9_batchnorm_layer')
            end_points[end_point] = network
        # conv10
        end_point = 'block10'
        with tf.variable_scope(end_point):
            network = tl.layers.Conv2dLayer(network, act=tf.nn.relu, shape=[1, 1, 256, 128], strides=[1, 1, 1, 1],
                                            padding='SAME', W_init=tf.glorot_uniform_initializer(), name='conv10_1')
            network = tl.layers.Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 128, 256], strides=[1, 1, 1, 1],
                                            padding='VALID', W_init=tf.glorot_uniform_initializer(),
                                            name='conv10_2')  # (-1,3,3,256)
            # network  = tl.layers.BatchNormLayer(network,is_train = is_training, act =tf.nn.relu,name='block10_batchnorm_layer')
            end_points[end_point] = network
        # conv11
        end_point = 'block11'
        with tf.variable_scope(end_point):
            network = tl.layers.Conv2dLayer(network, act=tf.nn.relu, shape=[1, 1, 256, 128], strides=[1, 1, 1, 1],
                                            padding='SAME', W_init=tf.glorot_uniform_initializer(), name='conv11_1')
            network = tl.layers.Conv2dLayer(network, act=tf.nn.relu, shape=[3, 3, 128, 256], strides=[1, 1, 1, 1],
                                            padding='VALID', W_init=tf.glorot_uniform_initializer(),
                                            name='conv11_2')  # (-1,1,1,256)
            end_points[end_point] = network
        predictions = []
        localisations = []
        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box'):
                cls_pred, loc_pred = ssd_multibox_layer(end_points[layer],
                                                        all_num_classes,
                                                        default_box_size[i],
                                                        normalizations[i])
            predictions.append(cls_pred.outputs)
            localisations.append(loc_pred.outputs)
        predictions = tf.concat(predictions, axis=1)
        localisations = tf.concat(localisations, axis=1)
        print('##   feature_class shape : ' + str(predictions.get_shape().as_list()))
        print('##   feature_location shape : ' + str(localisations.get_shape().as_list()))
        return predictions, localisations,end_points

def smooth_L1(x):
    return tf.where(tf.less_equal(tf.abs(x), 1.0), tf.multiply(0.5, tf.pow(x, 2.0)), tf.subtract(tf.abs(x), 0.5))


def mixedloss(feature_class, feature_location, groundtruth_class, groundtruth_location, groundtruth_positives,
              groundtruth_count):
    softmax_cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=feature_class,
                                                                           labels=groundtruth_class)
    loss_location = tf.div(tf.reduce_sum(tf.multiply(
        tf.reduce_sum(smooth_L1(tf.subtract(feature_location,groundtruth_location)),
                      reduction_indices=2), groundtruth_positives), reduction_indices=1),
        tf.reduce_sum(groundtruth_positives, reduction_indices=1))
    loss_class = tf.div(tf.reduce_sum(tf.multiply(softmax_cross_entropy, groundtruth_count), reduction_indices=1),tf.reduce_sum(groundtruth_positives, reduction_indices=1))

    loss_all = tf.reduce_sum(tf.add(loss_class, loss_location * 1))
    loss_classa = tf.reduce_sum(loss_class)
    loss_loca = tf.reduce_sum(loss_location)
    return loss_all, loss_classa, loss_loca


# def decodeFeature(default_bbox, feature_loction):
#     feature_loction_len = len(feature_loction)
#     for img_index in range(feature_loction_len):
#         for pre_actual in feature_loction[img_index]:
#             for boxe_index in range(all_default_boxs_len):
#                 cx = pre_actual[0] * all_default_boxs[boxe_index][2] + all_default_boxs[boxe_index][0]
#                 cy = pre_actual[1] * all_default_boxs[boxe_index][3] + all_default_boxs[boxe_index][1]
#                 feat_w = math.exp(pre_actual[2]) * all_default_boxs[boxe_index][2]
#                 feat_h = math.exp(pre_actual[3]) * all_default_boxs[boxe_index][3]
#                 feat_size = tl.prepro.obj_box_coord_scale_to_pixelunit([cx, cy, feat_w, feat_h], shape=(300, 300, 3))


if __name__ == '__main__':
    all_default_boxs = generate_all_default_boxs()
    all_default_boxs_len = len(all_default_boxs)
    imageinput = tf.placeholder(tf.float32, [None, 300, 300, 3], "inputsimage")
    groundtruth_class = tf.placeholder(shape=[None, all_default_boxs_len], dtype=tf.int32,
                                       name='groundtruth_class')
    groundtruth_location = tf.placeholder(shape=[None, all_default_boxs_len, 4], dtype=tf.float32,
                                          name='groundtruth_location')
    groundtruth_positives = tf.placeholder(shape=[None, all_default_boxs_len], dtype=tf.float32,
                                           name='groundtruth_positives')
    groundtruth_negatives = tf.placeholder(shape=[None, all_default_boxs_len], dtype=tf.float32,
                                           name='groundtruth_negatives')
    groundtruth_count = tf.add(groundtruth_positives, groundtruth_negatives)
    pre_cls, pre_loc,net_work = ssd_net(imageinput, True, False)
    loss_all, loss_class, loss_loc = mixedloss(pre_cls, pre_loc, groundtruth_class, groundtruth_location,
                                               groundtruth_positives, groundtruth_count)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_all)
    sess = tf.Session()

    sess.run(tf.global_variables_initializer())
    maybe_download_and_extract(
        'vgg16_weights.npz', 'models', 'http://www.cs.toronto.edu/~frossard/vgg16/', expected_bytes=553436134
    )
    npz = np.load(os.path.join('models', 'vgg16_weights.npz'))
    params = []
    idx  = 0
    for val in sorted(npz.items()):
        idx=idx+1
        print("  Loading params %s"%(str(val[1].shape)))
        params.append(val[1])
        if idx==17:
            break
    assign_params(sess, params,net_work['block4'])
    for epoch in range(train_epoch):
        idx = 0
        for batch_imgs, batch_ann in tl.iterate.minibatches(laod_imgs, load_ann, batch_size, shuffle=False):
            g_img, g_ann = getTrainData(batch_imgs, batch_ann)
            gt_class, gt_location, gt_positives, gt_negatives = generate_groundtruth_data(g_ann)
            # print("g_img",g_img)
            # print("gt_positives",np.sum(gt_positives,axis=1))
            # print("gt_positives", np.sum(gt_negatives, axis=1))
            # print("pre_cls", sess.run(pre_cls, feed_dict={imageinput: g_img, groundtruth_class: gt_class,
            #                                               groundtruth_location: gt_location,
            #                                               groundtruth_positives: gt_positives,
            #                                               groundtruth_negatives: gt_negatives}))
            sess.run(train_op,feed_dict={imageinput: g_img, groundtruth_class: gt_class, groundtruth_location: gt_location,
                                groundtruth_positives: gt_positives, groundtruth_negatives: gt_negatives})
            # print("groundtruth_count",sess.run(sum_gt_count,feed_dict={groundtruth_positives:gt_positives,groundtruth_negatives:gt_negatives}))
            if idx % 1 == 0:
                loss, loss_clsassa, loss_loca = sess.run([loss_all, loss_class, loss_loc],
                                                         feed_dict={imageinput: g_img, groundtruth_class: gt_class,
                                                                    groundtruth_location: gt_location,
                                                                    groundtruth_positives: gt_positives,
                                                                    groundtruth_negatives: gt_negatives})
                print("idx:%d,epoch:%d,loss:%.4f,loss_classa:%.4f,loss_loca:%.4f," % (idx, epoch, loss, loss_clsassa, loss_loca))
            idx = idx + 1










