from .base_environment import BaseEnvironment
from tfne.helper_functions import read_option_from_config
from tfne.helper_functions import load_chexpert_data
from tfne.helper_functions import set_tensorflow_memory_growth
from tqdm.keras import TqdmCallback

import tensorflow as tf
import numpy as np

class CheXpertEnvironment(BaseEnvironment):
    def __init__(self, weight_training, config=None, verbosity=0, **kwargs):                                                                      
        print("Setting up CheXpert environment...")                                                                                                               
        # Load and preprocess CheXpert data here                                                                                 
        self.train_dataset, self.test_dataset = load_chexpert_data()                                 
        print("Data loaded")
        
        #self.accuracy_metric = tf.keras.metrics.AUC(multi_label=True)  # Using AUC for multi-label classification             
        self.accuracy_metric = tf.keras.metrics.BinaryAccuracy(threshold=0.5)
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
        
        #set_tensorflow_memory_growth()
        # Pull a single batch from the training dataset
        #for image_batch, label_batch in self.train_dataset.take(1):
        #    print("Image batch shape: ", image_batch.shape)
        #    print("Label batch shape: ", label_batch.shape)
        #    # Print the first few values of the first image in the batch
        #    print("First few pixel values of the first image in batch: ", image_batch[0].numpy()[0][0])
        #    # Print the first label in the batch
        #    print("First label in batch:", label_batch[0].numpy())
        try:
            model = genome.get_model()                                                                                                                                    
            model.summary()
            optimizer = genome.get_optimizer()                                                                                                                          
            print('model compiling,...........')
            model.compile(optimizer=optimizer,                                                                                                                          
                          loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),                                                                      
                          metrics=[self.accuracy_metric])                                                                                                                   
            print('Starting model training,...............')
            model.fit(self.train_dataset,
                      epochs=self.epochs,                                                                                                                                        
                      verbose=2, callbacks=[TqdmCallback(verbose=2)])                                                                                                                                    

            print('Model training complete,...............')
            #predictions = model.predict(self.test_dataset)

            # Check for NaNs in predictions
                       # Determine fitness by updating the accuracy metric
            # Iterate over the test dataset batch by batch
            i = 0
            for image_batch, label_batch in self.test_dataset:
                #print('In for loop : ')
                #print(f"Shape of label_batch1: {label_batch.shape}")
                #print(f"Shape of image_batch1: {image_batch.shape}")
                i = i+1
                print('batch number : ', i)
                # Make predictions using your model
                # Assume 'model' is your trained model
                predictions = model.predict_on_batch(image_batch)
                # Update the state of the accuracy metric with the current batch
                #print("past prediction ,...........")
                self.accuracy_metric.update_state(label_batch, predictions)
                #print(f"Shape of predictions: {predictions.shape}")
                #print(f"Shape of label_batch: {label_batch.shape}")
           
            #if np.any(np.isnan(predictions)):
            #    print("Encountered NaN in predictions. Setting fitness to a low value.")
            #    return -1.0

            # After iterating over the test dataset and updating metric states
            final_accuracy = self.accuracy_metric.result().numpy()

            # Return the final_accuracy, rounded and multiplied by 100
            return round(final_accuracy * 100, 4)


        
        except Exception as e:
            # Handle exceptions that occurred during training or evaluation
            print(f"An exception occurred during training or evaluation: {e}")
            
            return -1.0  # Set a low fitness score
        
    def replay_genome(self, genome):
        print(f"Replaying Genome #{genome.get_id()}")
        model = genome.get_model()
        self.accuracy_metric.reset_states()
        
        # Iterate over the test dataset batch by batch
        for image_batch, label_batch in self.test_dataset:
            # Make predictions using your model
            # Assume 'model' is your trained model
            predictions = model.predict(image_batch)
            # Update the state of the accuracy metric with the current batch
            self.accuracy_metric.update_state(label_batch, np.argmax(predictions, axis=-1))
        
        # After iterating over the test dataset and updating metric states
        final_accuracy = self.accuracy_metric.result().numpy()

        #self.accuracy_metric.update_state(self.test_labels, model.predict(self.test_images))
        evaluated_fitness = round(final_accuracy * 100, 4)
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

