from __future__ import annotations

import math
import random
import statistics

import numpy as np
import tensorflow as tf

from .codeepneat_module_base import CoDeepNEATModuleBase
from tfne.helper_functions import round_with_step
from tensorflow.keras.layers import Layer

import tensorflow.compat.v2 as tf

from keras import backend
from keras.applications import imagenet_utils
from keras.engine import training
from keras.layers import VersionAwareLayers
from keras.utils import data_utils
from keras.utils import layer_utils

# isort: off
from tensorflow.python.util.tf_export import keras_export
from tensorflow.keras import Model, Input, layers, backend
layers = VersionAwareLayers()


def dense_block(x, blocks, growth_rate):
    """A dense block.

    Args:
      x: input tensor.
      blocks: integer, the number of building blocks.
      name: string, block label.

    Returns:
      Output tensor for the block.
    """
    for i in range(blocks):
        x = conv_block(x,growth_rate)
    return x


def transition_block(x, reduction):
    """A transition block.

    Args:
      x: input tensor.
      reduction: float, compression rate at transition layers.
      name: string, block label.

    Returns:
      output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5
    )(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(
        int(backend.int_shape(x)[bn_axis] * reduction),
        1,
        use_bias=False,
    )(x)
    x = layers.AveragePooling2D(2, strides=2)(x)
    return x


def conv_block(x,growth_rate):
    """A building block for a dense block.

    Args:
      x: input tensor.
      growth_rate: float, growth rate at dense layers.
      name: string, block label.

    Returns:
      Output tensor for the block.
    """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
    x1 = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5
    )(x)
    x1 = layers.Activation("relu")(x1)
    x1 = layers.Conv2D(
        4 * growth_rate, 1, use_bias=False
    )(x1)
    x1 = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5
    )(x1)
    x1 = layers.Activation("relu")(x1)
    x1 = layers.Conv2D(
        growth_rate, 3, padding="same", use_bias=False
    )(x1)
    x = layers.Concatenate(axis=bn_axis)([x, x1])
    return x

class SimAMModule(Layer):
    def __init__(self, e_lambda=1e-4, **kwargs):
        super(SimAMModule, self).__init__(**kwargs)
        self.e_lambda = e_lambda

    def build(self, input_shape):
        super(SimAMModule, self).build(input_shape)

    def call(self, x):
        # Calculating the dimensions
        b, h, w, c = x.shape

        n = h * w - 1

        # Subtracting the mean and squaring
        x_minus_mu_square = tf.math.square(x - tf.reduce_mean(x, axis=[1, 2], keepdims=True))

        # Applying the SimAM formula
        y = x_minus_mu_square / (4 * (tf.reduce_sum(x_minus_mu_square, axis=[1, 2], keepdims=True) / n + self.e_lambda)) + 0.5

        # Applying the sigmoid function
        y = tf.sigmoid(y)

        # Multiplying with the input
        return x * y

    def get_config(self):
        config = super(SimAMModule, self).get_config()
        config.update({
            'e_lambda': self.e_lambda
        })
        return config

    @staticmethod
    def get_module_name():
        return "simam"

class CoDeepNEATModuleDenseBlock(CoDeepNEATModuleBase):
    
    def __init__(self, 
                 config_params, 
                 module_id, 
                 parent_mutation, 
                 dtype,
                 filters=None,
                 num_layers=1,
                 num_dense_blocks=1,
                 growth_rate=None,
                 reduction_rate=None,
                 activation=None,
                 kernel_init=None,
                 kernel_size=None,
                 bias_init=None,
                 dropout_flag=None,
                 dropout_rate=None,
                 attention_module_flag=None,
                 transition_layer_flag=None,
                 batch_norm=None,
                 avg_pool_flag=None,
                 self_initialization_flag=False):
        """
        Initialize a DenseBlock module
        :param config_params: Dictionary containing configuration parameters
        :param module_id: Unique ID of this module
        :param parent_mutation: Dictionary that describes how this module was created
        :param dtype: Data type for TensorFlow (could be float32, float16, etc.)
        :param num_layers: Number of layers in the dense block
        :param growth_rate: The growth rate, which is the number of feature maps produced by each layer in the dense block
        :param activation: Activation function to use
        :param kernel_init: Initializer for the kernel weights
        :param bias_init: Initializer for the bias
        :param dropout_flag: Boolean, whether to use dropout
        :param dropout_rate: The rate for dropout
        """
        self.config_params = config_params
        self.module_id = module_id
        self.parent_mutation = parent_mutation
        self.dtype = dtype
        self.filters = filters
        self.num_layers = num_layers
        self.num_dense_blocks = num_dense_blocks
        self.growth_rate = growth_rate
        self.reduction_rate = reduction_rate
        self.activation = activation
        self.kernel_init = kernel_init
        self.kernel_size = kernel_size
        self.bias_init = bias_init
        self.dropout_flag = dropout_flag
        self.dropout_rate = dropout_rate
        self.attention_module_flag = attention_module_flag
        self.transition_layer_flag = transition_layer_flag
        self.batch_norm = batch_norm
        self.avg_pool_flag = avg_pool_flag
        
        # Register the implementation specifics by calling parent class
        super().__init__(config_params, module_id, parent_mutation, dtype)
        
        # If self_initialization_flag is provided, initialize the module parameters
        if self_initialization_flag:
            self._initialize()

    def __str__(self):
        """
        Return a string representation of the DenseBlock module.
        """
        return (f"Module ID: {self.module_id}, Type: DenseBlock, Fitness: {self.fitness}, "
                f"Num Layers: {self.num_layers}, Growth Rate: {self.growth_rate}, Reduction Rate: {self.reduction_rate} "
                f"Kernel Size: {self.kernel_size}, Activation: {self.activation}, "
                f"Batch Norm: {self.batch_norm}, Dropout Rate: {self.dropout_rate}")

    
    def _initialize(self):
        """
        Initialize the dense block module with the range of parameters specified in the config file.
        The parameters are randomly initialized within the specified range.
        """
    
        # Initialize growth_rate
        self.growth_rate = random.randint(self.config_params['growth_rate']['min'], 
                                      self.config_params['growth_rate']['max']) 
        # Initialize reduction_rate
        self.reduction_rate = random.uniform(self.config_params['reduction_rate']['min'], 
                                      self.config_params['reduction_rate']['max'])
        # Initialize num_layers
        self.num_layers = random.randint(self.config_params['num_layers']['min'], 
                                     self.config_params['num_layers']['max'])
    
        # Initialize activation
        self.activation = random.choice(self.config_params['activation'])
    
        # Initialize other attributes, like self.fitness, self.module_id, etc., 
        # based on your existing setup, typically via calling the superclass constructor.
        self.kernel_init = random.choice(self.config_params['kernel_init'])
        self.bias_init = random.choice(self.config_params['bias_init'])
        self.dropout_flag = random.random() < self.config_params['dropout_flag']
        self.attention_module_flag = random.random() < self.config_params['attention_module_flag']
        self.transition_layer_flag = random.random() < self.config_params['transition_layer_flag']
        self.avg_pool_flag = random.random() < self.config_params['avg_pool_flag']
        self.batch_norm = random.random() < self.config_params['batch_norm']
        self.dropout_rate = random.uniform(self.config_params['dropout_rate']['min'],
                                             self.config_params['dropout_rate']['max'])
    
        random_filters = random.randint(self.config_params['filters']['min'],
                                        self.config_params['filters']['max'])
        self.filters = round_with_step(random_filters,
                                       self.config_params['filters']['min'],
                                       self.config_params['filters']['max'],
                                       self.config_params['filters']['step'])

    def create_downsampling_layer(self, in_shape, out_shape) -> tf.keras.layers.Layer:
        """"""
        raise NotImplementedError("Downsampling has not yet been implemented for DenseDropout Modules")

    def create_module_layers(self):

       # Initialize empty list of layers
        module_layers = list()
       
        print('growth rate: ', self.growth_rate)
        print('reduction rate: ', self.reduction_rate)
        print('num layers: ', self.num_layers)
        print('attention mod flag : ', self.attention_module_flag) 
        print('transition layer flag : ', self.transition_layer_flag)

        #input_tensor = Input(shape=(None,None,None))

        #db = dense_block(input_tensor,self.num_layers,self.growth_rate)
        #if self.attention_module_flag:
        #    am = SimAMModule()
        #if self.transition_layer_flag:
        #    tl = transition_layer(input_tensor,0.5)

        #if self.attention_module_flag:
        #    module_layers.append(am)
        
        #if self.transition_layer_flag:
        #    module_layers.append(tl)
        x1 = layers.BatchNormalization(
        axis=3, epsilon=1.001e-5
        )
        x2 = layers.Activation("relu")(x1)
        x3 = layers.Conv2D(4 * self.growth_rate, 1, use_bias=False)(x2)
        x4 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x3)
        x5 = layers.Activation("relu")(x4)
        x6 = layers.Conv2D(self.growth_rate, 3, padding="same", use_bias=False)(x5)
        #x = layers.Concatenate(axis=bn_axis)([x, x1])
        
        return x6

    def create_mutation(self, offspring_id, max_degree_of_mutation):
        # Copy the parameters of this parent module for the parameters of the offspring
        offspring_params = {'num_layers': self.num_layers,
                            'growth_rate': self.growth_rate,
                            'reduction_rate': self.reduction_rate,
                            'attention_module_flag': self.attention_module_flag,
                            'transition_layer_flag': self.transition_layer_flag}
    
        # Create the dict that keeps track of the mutations occurring for the offspring
        parent_mutation = {'parent_id': self.module_id,
                           'mutation': 'mutation',
                           'mutated_params': dict()}
    
        # Determine the number of parameters to be mutated
        param_mutation_count = math.ceil(max_degree_of_mutation * len(offspring_params))
    
        # Uniformly randomly choose the parameters to be mutated
        parameters_to_mutate = random.sample(list(offspring_params.keys()), k=param_mutation_count)
    
        for param_to_mutate in parameters_to_mutate:
            if param_to_mutate == 'num_layers':
                perturbed_layers = int(np.random.normal(loc=self.num_layers, scale=self.config_params['num_layers']['stddev']))
                offspring_params['num_layers'] = max(self.config_params['num_layers']['min'], 
                                                     min(self.config_params['num_layers']['max'], perturbed_layers))
                parent_mutation['mutated_params']['num_layers'] = self.num_layers
                
            elif param_to_mutate == 'growth_rate':
                perturbed_growth_rate = np.random.normal(loc=self.growth_rate, 
                                                         scale=self.config_params['growth_rate']['stddev'])
                offspring_params['growth_rate'] = max(self.config_params['growth_rate']['min'], 
                                                      min(self.config_params['growth_rate']['max'], perturbed_growth_rate))
                parent_mutation['mutated_params']['growth_rate'] = self.growth_rate
    
            elif param_to_mutate == 'reduction_rate':
                perturbed_reduction_rate = np.random.normal(loc=self.reduction_rate, 
                                                         scale=self.config_params['reduction_rate']['stddev'])
                offspring_params['reduction_rate'] = max(self.config_params['reduction_rate']['min'], 
                                                      min(self.config_params['reduction_rate']['max'], perturbed_reduction_rate))
                parent_mutation['mutated_params']['reduction_rate'] = self.reduction_rate
    
            elif param_to_mutate == 'transition_layer_flag':
                offspring_params['transition_layer_flag'] = not self.transition_layer_flag
                parent_mutation['mutated_params']['transition_layer_flag'] = self.transition_layer_flag
    
            elif param_to_mutate == 'attention_module_flag':
                offspring_params['attention_module_flag'] = not self.attention_module_flag
                parent_mutation['mutated_params']['attention_module_flag'] = self.attention_module_flag
    
        return CoDeepNEATModuleDenseBlock(config_params=self.config_params,
                                          module_id=offspring_id,
                                          parent_mutation=parent_mutation,
                                          dtype=self.dtype,
                                          **offspring_params)

    def create_crossover(self,
                         offspring_id,
                         less_fit_module,
                         max_degree_of_mutation) -> CoDeepNEATModuleDenseBlock:
        """
        Create crossed over DenseBlock module and return it. Carry over parameters of fitter parent for categorical
        parameters and calculate parameter average between both modules for sortable parameters
        @param offspring_id: int of unique module ID of the offspring
        @param less_fit_module: second DenseBlock module with lower fitness
        @param max_degree_of_mutation: float between 0 and 1 specifying the maximum degree of mutation
        @return: instantiated DenseBlock module with crossed over parameters
        """
        # Create offspring parameters by carrying over parameters of fitter parent for categorical parameters and
        # calculating parameter average between both modules for sortable parameters
        offspring_params = dict()
    
        # Create the dict that keeps track of the mutations occurring for the offspring
        parent_mutation = {'parent_id': (self.module_id, less_fit_module.get_id()),
                           'mutation': 'crossover'}
    
        offspring_params['num_layers'] = round_with_step(int((self.num_layers + less_fit_module.num_layers) / 2),
                                                         self.config_params['num_layers']['min'],
                                                         self.config_params['num_layers']['max'],
                                                         self.config_params['num_layers']['step'])
    
        offspring_params['growth_rate'] = round_with_step((self.growth_rate + less_fit_module.growth_rate) / 2,
                                                          self.config_params['growth_rate']['min'],
                                                          self.config_params['growth_rate']['max'],
                                                          self.config_params['growth_rate']['step'])
   
        offspring_params['reduction_rate'] = round_with_step((self.reduction_rate + less_fit_module.reduction_rate) / 2,
                                                          self.config_params['reduction_rate']['min'],
                                                          self.config_params['reduction_rate']['max'],
                                                          self.config_params['reduction_rate']['step'])

        # Assuming that categorical features like batch normalization or dropout are either enabled or disabled
        offspring_params['transition_layer_flag'] = self.transition_layer_flag
    
        # Assuming that categorical features like batch normalization or dropout are either enabled or disabled
        offspring_params['attention_module_flag'] = self.attention_module_flag 

        return CoDeepNEATModuleDenseBlock(config_params=self.config_params,
                                          module_id=offspring_id,
                                          parent_mutation=parent_mutation,
                                          dtype=self.dtype,
                                          **offspring_params)
    
    def serialize(self) -> dict:
        """
        @return: serialized constructor variables of the module as json-compatible dict
        """
        return {
            'module_type': self.get_module_type(),
            'module_id': self.module_id,
            'parent_mutation': self.parent_mutation,
            'num_layers': self.num_layers,
            'growth_rate': self.growth_rate,
            'activation': self.activation,
            'kernel_init': self.kernel_init,
            'bias_init': self.bias_init,
            'dropout_flag': self.dropout_flag,
            'dropout_rate': self.dropout_rate,
            'batch_norm': self.batch_norm  # if you have this parameter
        }

    def get_distance(self, other_module) -> float:
        """
        Calculate distance between 2 DenseDropout modules by inspecting each parameter, calculating the congruence
        between each and eventually averaging the out the congruence. The distance is returned as the average
        congruences distance to 1.0. The congruence of continuous parameters is calculated by their relative distance.
        The congruence of categorical parameters is either 1.0 in case they are the same or it's 1 divided to the amount
        of possible values for that specific parameter. Return the calculated distance.
        @param other_module: second DenseDropout module to which the distance has to be calculated
        @return: float between 0 and 1. High values indicating difference, low values indicating similarity
        """
        congruence_list = list()
        if self.merge_method == other_module.merge_method:
            congruence_list.append(1.0)
        else:
            congruence_list.append(1 / len(self.config_params['merge_method']))
        if self.units >= other_module.units:
            congruence_list.append(other_module.units / self.units)
        else:
            congruence_list.append(self.units / other_module.units)
        if self.activation == other_module.activation:
            congruence_list.append(1.0)
        else:
            congruence_list.append(1 / len(self.config_params['activation']))
        if self.kernel_init == other_module.kernel_init:
            congruence_list.append(1.0)
        else:
            congruence_list.append(1 / len(self.config_params['kernel_init']))
        if self.bias_init == other_module.bias_init:
            congruence_list.append(1.0)
        else:
            congruence_list.append(1 / len(self.config_params['bias_init']))
        congruence_list.append(abs(self.dropout_flag - other_module.dropout_flag))
        if self.dropout_rate >= other_module.dropout_rate:
            congruence_list.append(other_module.dropout_rate / self.dropout_rate)
        else:
            congruence_list.append(self.dropout_rate / other_module.dropout_rate)

        congruence_list.append(abs(self.batch_norm - other_module.batch_norm))
    
        # Return the distance as the distance of the average congruence to the perfect congruence of 1.0
        return round(1.0 - statistics.mean(congruence_list), 4)

    def get_module_type(self) -> str:
        """"""
        return 'DenseBlock'
