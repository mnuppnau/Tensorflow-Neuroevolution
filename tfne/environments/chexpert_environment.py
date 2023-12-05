from .base_environment import BaseEnvironment
from tensorflow.keras.metrics import BinaryAccuracy,AUC
from tensorflow.keras.optimizers import Adam
from tfne.helper_functions import read_option_from_config
from tfne.helper_functions import load_chexpert_data
from tfne.helper_functions import set_tensorflow_memory_growth
from tqdm.keras import TqdmCallback
from tensorflow.keras.callbacks import EarlyStopping

import tensorflow as tf
import numpy as np

class CheXpertEnvironment(BaseEnvironment):
    def __init__(self, weight_training, config=None, verbosity=0, **kwargs):                                                                      
        print("Setting up CheXpert environment...")                                                                                                               
        # Load and preprocess CheXpert data here                                                                                 
        self.train_dataset, self.test_dataset = load_chexpert_data()                                 
        print("Data loaded")
        
        self.accuracy_metric = tf.keras.metrics.AUC(curve='PR',multi_label=True)  # Using AUC for multi-label classification             
        #self.accuracy_metric = tf.keras.metrics.BinaryAccuracy()
        #self.accuracy_list = ['binary_accuracy','accuracy',AUC(name='auc',multi_label=True)]
        self.verbosity = verbosity                                                                                                                                           

        if not weight_training:                                                                                                                                                 
            raise NotImplementedError("Non-weight training evaluation not yet implemented for CheXpert environment")   

        elif config is None and len(kwargs) == 0:                                                                                                                  
            raise RuntimeError("Neither config file nor explicit config parameters for weight training were supplied")           

        elif len(kwargs) == 0:                                                                                                                                                 
            self.eval_genome_fitness = self._eval_genome_fitness_weight_training                                                             
            self.epochs = read_option_from_config(config, 'EVALUATION', 'epochs')                                                            
            #self.batch_size = read_option_from_config(config, 'EVALUATION', 'batch_size')                                                 
            
            # Apply batch and prefetch to the dataset
            #self.train_dataset = self.train_dataset.batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            #self.test_dataset = self.test_dataset.batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            
        elif config is None:                                                                                                                                                     
            self.eval_genome_fitness = self._eval_genome_fitness_weight_training                                                             
            self.epochs = kwargs['epochs']                                                                                                                              
            #self.batch_size = kwargs['batch_size'] 
        
            # Apply batch and prefetch to the dataset
            #self.train_dataset = self.train_dataset.batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            #self.test_dataset = self.test_dataset.batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def eval_genome_fitness(self, genome) -> float:
        # TO BE OVERRIDEN
        raise RuntimeError()

    def _eval_genome_fitness_weight_training(self, genome) -> float:
        try:
            model = genome.get_model()                                                                                                                                    
            model.summary()
            #optimizer = genome.get_optimizer()                                                                                                                          
            optimizer = Adam(learning_rate = 0.0001)
            #optimizer = mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)
            print('model compiling,...........')
            model.compile(optimizer=optimizer,                                                                                                                          
                          loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),                                                                      
                          metrics=[self.accuracy_metric])                                                                                                                   
            print('Starting model training,...............')
            try:
                model.fit(self.train_dataset,
                      epochs=self.epochs,                                                                                                                                        
                      verbose=2)
            except Exception as e:
                print(f"An error occurred during training: {e}")
                return -1

            print('Model training complete,...............')
            #predictions = model.predict(self.test_dataset)
            self.accuracy_metric.reset_state()
            # Check for NaNs in predictions
                       # Determine fitness by updating the accuracy metric
            # Iterate over the test dataset batch by batch
            #i = 0
            #for image_batch, label_batch in self.test_dataset:
            #    predictions = model.predict_on_batch(image_batch)
            #    self.accuracy_metric.update_state(label_batch, predictions)
           
                # Iterate over the test dataset in batches and update metric states
            for batch_images, batch_labels in self.test_dataset:
                predictions = model(batch_images, training=False)
                self.accuracy_metric.update_state(batch_labels, predictions)

            # After iterating over the test dataset and updating metric states
            final_accuracy = self.accuracy_metric.result().numpy()

            # Return the final_accuracy, rounded and multiplied by 100
            return float(round(final_accuracy * 100, 4))
        
        except Exception as e:
            # Handle exceptions that occurred during training or evaluation
            print(f"An exception occurred during training or evaluation: {e}")
            
            return -1.0  # Set a low fitness score
        
    def replay_genome(self, genome):
        print(f"Replaying Genome #{genome.get_id()}")
        model = genome.get_model()
        self.accuracy_metric.reset_state()
        
        # Iterate over the test dataset batch by batch
        #for image_batch, label_batch in self.test_dataset:
            # Make predictions using your model
            # Assume 'model' is your trained model
        #    predictions = model.predict(image_batch)
            # Update the state of the accuracy metric with the current batch
        #    self.accuracy_metric.update_state(label_batch, np.argmax(predictions, axis=-1))
           # Iterate over the test dataset in batches and update metric states
        
        for batch_images, batch_labels in self.test_dataset:
            predictions = model(batch_images, training=False)
            self.accuracy_metric.update_state(batch_labels, predictions)
 
        # After iterating over the test dataset and updating metric states
        final_accuracy = self.accuracy_metric.result().numpy()

        #self.accuracy_metric.update_state(self.test_labels, model.predict(self.test_images))
        evaluated_fitness = float(round(final_accuracy * 100,4))
        print(f"Achieved Fitness: {evaluated_fitness}\n")

    def duplicate(self) -> 'CheXpertEnvironment':
        if hasattr(self, 'epochs'):
            return CheXpertEnvironment(True, verbosity=self.verbosity, epochs=self.epochs, batch_size=self.batch_size)
        else:
            return CheXpertEnvironment(False, verbosity=self.verbosity)

    def get_input_shape(self) -> (int, int, int):
        return 224, 224, 3  # Replace with the actual dimensions

    def get_output_shape(self) -> (int,):
        return (5,)  # 14 tasks in CheXpert multi-label classification

