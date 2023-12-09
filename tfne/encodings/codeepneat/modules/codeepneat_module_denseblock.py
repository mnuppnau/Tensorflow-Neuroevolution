from __future__ import annotations

import math
import random
import statistics
import tensorflow.compat.v2 as tf
import numpy as np
import tensorflow as tf

from absl import logging
from .codeepneat_module_base import CoDeepNEATModuleBase
from tfne.helper_functions import round_with_step, select_random_value, SimAMModule
from keras import backend
from keras.applications import imagenet_utils
from keras.engine import training
from keras.layers import VersionAwareLayers
from keras.utils import data_utils
from keras.utils import layer_utils

layers = VersionAwareLayers()

logging.use_absl_handler()

logging.get_absl_handler().use_absl_log_file('absl_logging', './logs') 
#absl.flags.FLAGS.mark_as_parsed() 
logging.set_verbosity(logging.INFO)

class CoDeepNEATModuleDenseBlock(CoDeepNEATModuleBase):
    
    def __init__(self, 
                 config_params, 
                 module_id, 
                 parent_mutation, 
                 dtype,
                 merge_method=None,
                 num_layers=4,
                 growth_rate=None,
                 reduction_rate=None,
                 include_simam=None,
                 simam_placed_in_db=None,
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
        self.merge_method = merge_method
        self.num_layers = num_layers
        self.growth_rate = growth_rate
        self.reduction_rate = reduction_rate
        self.include_simam = include_simam
        self.simam_placed_in_db = simam_placed_in_db

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
                f"Num Layers: {self.num_layers}, Growth Rate: {self.growth_rate}, "
                f"Reduction Rate: {self.reduction_rate}, SimAM Placed in DB: {self.simam_placed_in_db}")

    def _initialize(self):
        """
        Initialize the dense block module with the range of parameters specified in the config file.
        The parameters are randomly initialized within the specified range.
        """
    
        # Initialize growth_rate
        self.growth_rate = select_random_value(self.config_params['growth_rate']['min'], 
                                               self.config_params['growth_rate']['max'],
                                               self.config_params['growth_rate']['step'],
                                               self.config_params['growth_rate']['stddev'])
   
        random_reduction_rate = random.uniform(self.config_params['reduction_rate']['min'],
                                             self.config_params['reduction_rate']['max'])
        
        self.reduction_rate = round_with_step(random_reduction_rate,
                                            self.config_params['reduction_rate']['min'],
                                            self.config_params['reduction_rate']['max'],
                                            self.config_params['reduction_rate']['step'])

        self.num_layers = select_random_value(self.config_params['num_layers']['min'], 
                                               self.config_params['num_layers']['max'],
                                               self.config_params['num_layers']['step'],
                                               self.config_params['num_layers']['stddev'])
     
        self.include_simam = random.random() < self.config_params['include_simam']
        
        self.simam_placed_in_db = random.random() < self.config_params['simam_placed_in_db']
        
        self.merge_method = random.choice(self.config_params['merge_method'])
        self.merge_method['config']['dtype'] = self.dtype 
        
        logging.info('growth rate is %d',self.growth_rate)
        logging.info('reduction rate is %f',self.reduction_rate)
        logging.info('number of layers is %d',self.num_layers)
        logging.info('simam placed inside dense block is %d',self.simam_placed_in_db)
        logging.info('include simam is %d',self.include_simam)

    def create_downsampling_layer(self, in_shape, out_shape) -> tf.keras.layers.Layer:
        """
        Create a downsampling layer to transform a tensor from in_shape to out_shape.
        @param in_shape: Tuple of input shape (batch, height, width, channels).
        @param out_shape: Tuple of desired output shape (batch, height, width, channels).
        @return: Instantiated TF Conv2D layer or pooling layer that can downsample in_shape to out_shape.
        """
        #if not (len(in_shape) == 4 and len(out_shape) == 4):
        #    raise NotImplementedError("Downsampling Layer for shapes not having 4 dimensions is not implemented.")
        print('in shape : ', in_shape)
        print('out_shape : ', out_shape)
        # Handling spatial dimensions (height and width) downsampling
        if out_shape[1] is not None and out_shape[2] is not None:
            # Compute the stride needed for downsampling
            stride_height = int(in_shape[1] / out_shape[1])
            stride_width = int(in_shape[2] / out_shape[2])
            filters = in_shape[3]  # Keep the same number of channels

            # Use a Conv2D layer with strides for downsampling
            return tf.keras.layers.Conv2D(filters=filters,
                                          kernel_size=(3, 3),
                                          strides=(stride_height, stride_width),
                                          padding='same',
                                          activation=None,
                                          dtype=self.dtype)

        # Handling channel dimension downsampling
        if out_shape[3] is not None:
            filters = out_shape[3]  # Adjust the number of channels
            return tf.keras.layers.Conv2D(filters=filters,
                                          kernel_size=(1, 1),
                                          strides=(1, 1),
                                          padding='same',
                                          activation=None,
                                          dtype=self.dtype)

        raise RuntimeError("Unsupported downsampling operation for the given input and output shapes.")

    def create_module_layers(self):

        def dense_block(input_tensor, num_layers=self.num_layers, growth_rate=self.growth_rate):
            concatenated_outputs = [input_tensor]

            for i in range(num_layers):
                x = input_tensor if i == 0 else tf.keras.layers.Concatenate(axis=-1)(concatenated_outputs)

                x1 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)
                x1 = layers.Activation("relu")(x1)
                x1 = layers.Conv2D(4 * self.growth_rate, 1, use_bias=False)(x1)
                x1 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x1)
                x1 = layers.Activation("relu")(x1)
                x1 = layers.Conv2D(self.growth_rate, 3, padding="same", use_bias=False)(x1)

                concatenated_outputs.append(x1)

            return tf.keras.layers.Concatenate(axis=-1)(concatenated_outputs)

        def transition_layer(input_tensor, reduction=self.reduction_rate):
            x = input_tensor
            
            x = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)
            x = layers.Activation("relu")(x)
            x = layers.Conv2D(int(backend.int_shape(x)[3] * reduction),1,use_bias=False,)(x)
            x = layers.AveragePooling2D(2, strides=2)(x)
            
            return x

        def dense_block_w_simam(input_tensor, num_layers=self.num_layers, growth_rate=self.growth_rate):
            concatenated_outputs = [input_tensor]

            for i in range(num_layers):
                x = input_tensor if i == 0 else tf.keras.layers.Concatenate(axis=-1)(concatenated_outputs)

                x1 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x)
                x1 = layers.Activation("relu")(x1)
                x1 = layers.Conv2D(4 * self.growth_rate, 1, use_bias=False)(x1)
                x1 = layers.BatchNormalization(axis=3, epsilon=1.001e-5)(x1)
                x1 = layers.Activation("relu")(x1)
                x1 = layers.Conv2D(self.growth_rate, 3, padding="same", use_bias=False)(x1)

                if i == num_layers-1:
                    x1 = SimAMModule()(x1)
                
                concatenated_outputs.append(x1)

            return tf.keras.layers.Concatenate(axis=-1)(concatenated_outputs)


        # Initialize empty list of layers
        module_layers = list()

        if self.include_simam and self.simam_placed_in_db:
            # Dense Block Module
            dense_module = lambda input_tensor: dense_block_w_simam(
                input_tensor,
                num_layers=self.num_layers,
                growth_rate=self.growth_rate
            )
        else:
            # Dense Block Module
            dense_module = lambda input_tensor: dense_block(
                input_tensor,
                num_layers=self.num_layers,
                growth_rate=self.growth_rate
            )

        module_layers.append(dense_module)

        if not self.simam_placed_in_db and self.include_simam:
            # SimAM Module
            simam_module = lambda input_tensor: SimAMModule()(input_tensor)
            module_layers.append(simam_module)

        # Transition Layer Module
        transition_module = lambda input_tensor: transition_layer(
            input_tensor,
            reduction=self.reduction_rate
        )

        module_layers.append(transition_module)

        return module_layers

    def create_mutation(self, offspring_id, max_degree_of_mutation):
        # Copy the parameters of this parent module for the parameters of the offspring
        offspring_params = {'merge_method': self.merge_method,
                            'num_layers': self.num_layers,
                            'growth_rate': self.growth_rate,
                            'reduction_rate': self.reduction_rate,
                            'include_simam': self.include_simam,
                            'simam_placed_in_db': self.simam_placed_in_db}
    
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
                perturbed_num_layers = int(np.random.normal(loc=self.num_layers, scale=self.config_params['num_layers']['stddev']))
                rounded_num_layers = round_with_step(perturbed_num_layers,
                                                              self.config_params['num_layers']['min'],
                                                              self.config_params['num_layers']['max'],
                                                              self.config_params['num_layers']['step']) 
               
                if rounded_num_layers < 4:
                    offspring_params['num_layers'] = 4
                else:
                    offspring_params['num_layers'] = rounded_num_layers
                
                parent_mutation['mutated_params']['num_layers'] = self.num_layers
                
            elif param_to_mutate == 'growth_rate':
                perturbed_growth_rate = int(np.random.normal(loc=self.growth_rate, scale=self.config_params['growth_rate']['stddev']))
                offspring_params['growth_rate'] = round_with_step(perturbed_growth_rate,
                                                              self.config_params['growth_rate']['min'],
                                                              self.config_params['growth_rate']['max'],
                                                              self.config_params['growth_rate']['step']) 
               
                parent_mutation['mutated_params']['growth_rate'] = self.growth_rate
    
            elif param_to_mutate == 'reduction_rate':
                perturbed_reduction_rate = np.random.normal(loc=self.reduction_rate,
                                                          scale=self.config_params['reduction_rate']['stddev'])
                
                offspring_params['reduction_rate'] = round_with_step(perturbed_reduction_rate,
                                                                   self.config_params['reduction_rate']['min'],
                                                                   self.config_params['reduction_rate']['max'],
                                                                   self.config_params['reduction_rate']['step'])

                parent_mutation['mutated_params']['reduction_rate'] = self.reduction_rate
    
            elif param_to_mutate == 'include_simam':
                offspring_params['include_simam'] = not self.include_simam
                parent_mutation['mutated_params']['include_simam'] = self.include_simam
   
            elif param_to_mutate == 'simam_placed_in_db':
                offspring_params['simam_placed_in_db'] = not self.simam_placed_in_db
                parent_mutation['mutated_params']['simam_placed_in_db'] = self.simam_placed_in_db
   
            elif param_to_mutate == 'merge_method':
                offspring_params['merge_method'] = self.merge_method
                parent_mutation['mutated_params']['merge_method'] = self.merge_method

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
    
        rounded_num_layers = round_with_step(int((self.num_layers + less_fit_module.num_layers) / 2),
                                                         self.config_params['num_layers']['min'],
                                                         self.config_params['num_layers']['max'],
                                                         self.config_params['num_layers']['step'])
    
        if rounded_num_layers < 4:
            offspring_params['num_layers'] = 4
        else:
            offspring_params['num_layers'] = rounded_num_layers
                
        offspring_params['growth_rate'] = round_with_step(int((self.growth_rate + less_fit_module.growth_rate) / 2),
                                                          self.config_params['growth_rate']['min'],
                                                          self.config_params['growth_rate']['max'],
                                                          self.config_params['growth_rate']['step'])
   
        offspring_params['reduction_rate'] = round_with_step((self.reduction_rate + less_fit_module.reduction_rate) / 2,
                                                          self.config_params['reduction_rate']['min'],
                                                          self.config_params['reduction_rate']['max'],
                                                          self.config_params['reduction_rate']['step'])
   
        offspring_params['include_simam'] = self.include_simam
   
        offspring_params['simam_placed_in_db'] = self.simam_placed_in_db
   
        offspring_params['merge_method'] = self.merge_method

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
            'merge_method': self.merge_method,
            'num_layers': self.num_layers,
            'growth_rate': self.growth_rate,
            'include_simam': self.include_simam,
            'simam_placed_in_db': self.simam_placed_in_db,
            'reduction_rate': self.reduction_rate # if you have this parameter
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
      
        if self.num_layers >= other_module.num_layers:
            congruence_list.append(other_module.num_layers / self.num_layers)
        else:
            congruence_list.append(self.num_layers / other_module.num_layers)

        if self.growth_rate >= other_module.growth_rate:
            congruence_list.append(other_module.growth_rate / self.growth_rate)
        else:
            congruence_list.append(self.growth_rate / other_module.growth_rate)

        if self.reduction_rate >= other_module.reduction_rate:
            congruence_list.append(other_module.reduction_rate / self.reduction_rate)
        else:
            congruence_list.append(self.reduction_rate / other_module.reduction_rate)

        #congruence_list.append(abs(self.simam_placed_in_db - other_module.simam_placed_in_db))
        congruence_list.append(1 if self.include_simam == other_module.include_simam else 0)
        # Return the distance as the distance of the average congruence to the perfect congruence of 1.0
        congruence_list.append(1 if self.simam_placed_in_db == other_module.simam_placed_in_db else 0) 

        return round(1.0 - statistics.mean(congruence_list), 4)

    def get_module_type(self) -> str:
        """"""
        return 'DenseBlock'
