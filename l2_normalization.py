import tensorflow as tf
import numpy as np
import tensorlayer as tl
from tensorlayer.layers.core import Layer
from tensorlayer.layers.core import LayersConfig
from tensorflow.python.ops import init_ops

class L2_normlization_Layer(Layer):
    def __init__(self, prev_layer=None, scale_init=init_ops.ones_initializer, is_train=True,act=None, name='l2_normalization_layer'):
        # 校验名字是否已被使用（不变）
        super(L2_normlization_Layer, self).__init__(prev_layer=prev_layer, act=act, name=name)
        # 本层输入是上层的输出（不变）
        self.inputs = prev_layer.outputs
        # 输出信息（自定义部分）
        print("L2_normlization_Layer %s: %s, %s" % (self.name, self.act.__name__ if self.act is not None else 'No Activation', is_train))
        # 本层的功能实现（自定义部分）
        with tf.variable_scope(name) as vs:
            variables = []
            inputs_shape = self.inputs.get_shape()#(-1,38,38,512)
            print("inputs_shape",inputs_shape)
            inputs_rank = inputs_shape.ndims#4
            print("inputs_rank", inputs_rank)
            dtype = self.inputs.dtype.base_dtype
            norm_dim = tf.range(inputs_rank - 1, inputs_rank)#(3,4)
            print("norm_dim", norm_dim)
            params_shape = inputs_shape[-1:]
            print("params_shape", params_shape)
            self.outputs = tf.nn.l2_normalize(self.inputs, norm_dim, epsilon=1e-12)
            if scale_init:
                if scale_init == init_ops.ones_initializer:
                    scale_init = scale_init()
                scale = tf.get_variable(
                    'scale', shape=params_shape, initializer=scale_init, dtype=LayersConfig.tf_dtype, trainable=is_train )
                variables.append(scale)
                if act:
                    self.outputs = act(tf.multiply(self.outputs, scale))
                else:
                    self.outputs = tf.multiply(self.outputs, scale)
            else:
                scale = None
        self._add_layers(self.outputs)
        self._add_params(variables)