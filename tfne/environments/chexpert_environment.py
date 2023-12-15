from .base_environment import BaseEnvironment
from tensorflow.keras.metrics import BinaryAccuracy,AUC
from tensorflow.keras.optimizers import Adam
from tfne.helper_functions import read_option_from_config, load_chexpert_data, set_tensorflow_memory_growth, strategy
from tqdm.keras import TqdmCallback
from tensorflow.keras.callbacks import EarlyStopping
from absl import logging

import tensorflow as tf
import numpy as np
import gc
import json
from datetime import datetime
import io
import os

class CheXpertEnvironment(BaseEnvironment):
    def __init__(self, weight_training, config=None, verbosity=0, **kwargs):                                                                      
        print("Setting up CheXpert environment...")                                                                                                               
        # Load and preprocess CheXpert data here                                                                                 
        self.train_dist_dataset, self.test_dist_dataset = load_chexpert_data(config=config)                                 
        print("Data loaded")
       
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
            self.batch_size = read_option_from_config(config, 'EVALUATION', 'batch_size')                                                 
            self.batch_size_per_replica = read_option_from_config(config, 'EVALUATION', 'batch_size_per_replica')                                                 

            # Apply batch and prefetch to the dataset
            #self.train_dataset = self.train_dataset.batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            #self.test_dataset = self.test_dataset.batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            
        elif config is None:                                                                                                                                                     
            self.eval_genome_fitness = self._eval_genome_fitness_weight_training                                                             
            self.epochs = kwargs['epochs']                                                                                                                              
            self.batch_size = kwargs['batch_size'] 
            self.batch_size_per_replica = kwargs['batch_size_per_replica'] 
            # Apply batch and prefetch to the dataset
            #self.train_dataset = self.train_dataset.batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
            #self.test_dataset = self.test_dataset.batch(self.batch_size).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    def save_model_data(self, model_summary, accuracy):
        # Ensure the logs directory exists
        logs_dir = '/tmp/logs/'
    
        # Generate a timestamped filename within the logs directory
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        filename = f"{logs_dir}/model_{timestamp}.json"
    
        # Model data
        data = {
            "summary": model_summary,
            "accuracy": accuracy
        }
    
        # Writing to JSON file
        with open(filename, 'w') as file:
            json.dump(data, file, indent=4)

    def eval_genome_fitness(self, genome) -> float:
        # TO BE OVERRIDEN
        raise RuntimeError()


    def _eval_genome_fitness_weight_training(self, genome) -> float:
        with strategy.scope():
            # Set reduction to `NONE` so you can do the reduction yourself.
            self.loss_object = tf.keras.losses.BinaryCrossentropy(
                from_logits=False,
                reduction=tf.keras.losses.Reduction.NONE)
            def compute_loss(labels, predictions, model_losses):
              per_example_loss = self.loss_object(labels, predictions)
              loss = tf.nn.compute_average_loss(per_example_loss)
              if model_losses:
                loss += tf.nn.scale_regularization_loss(tf.add_n(model_losses))
              return loss
        
            self.test_loss = tf.keras.metrics.Mean(name='test_loss')

            self.train_accuracy = tf.keras.metrics.AUC(curve='PR',multi_label=True,name='train_accuracy')  # Using AUC for multi-label classification             
            self.test_accuracy = tf.keras.metrics.AUC(curve='PR',multi_label=True,name='test_accuracy')  # Using AUC for multi-label classification             


            self.model = genome.get_model()
            optimizer = Adam(learning_rate = 0.0001)
            #checkpoint = tf.train.Checkpoint(optimizer=optimizer, self.model=model)

        def train_step(inputs):
            images, labels = inputs

            with tf.GradientTape() as tape:
                predictions = self.model(images, training=True)
                loss = compute_loss(labels, predictions, self.model.losses)
        
            gradients = tape.gradient(loss, self.model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
            self.train_accuracy.update_state(labels, predictions)
            return loss
        
        def test_step(inputs):
            images, labels = inputs
        
            predictions = self.model(images, training=False)
            t_loss = self.loss_object(labels, predictions)
        
            self.test_loss.update_state(t_loss)
            self.test_accuracy.update_state(labels, predictions)

        # `run` replicates the provided computation and runs it
        # with the distributed input.
        @tf.function
        def distributed_train_step(dataset_inputs):
          per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
          return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                 axis=None)
        
        @tf.function
        def distributed_test_step(dataset_inputs):
          return strategy.run(test_step, args=(dataset_inputs,))

        #logging.info(self.model.summary())
        
        # TRAIN LOOP
        total_loss = 0.0
        num_batches = 0
        for x in self.train_dist_dataset:
            total_loss += distributed_train_step(x)
            num_batches += 1
        train_loss = total_loss / num_batches
        
        # TEST LOOP
        for x in self.test_dist_dataset:
            distributed_test_step(x)
        
        #if epoch % 2 == 0:
        #  checkpoint.save(checkpoint_prefix)
        
        template = ("Loss: {}, Accuracy: {}, Test Loss: {}, "
                      "Test Accuracy: {}")
        print(template.format( train_loss,
                                 self.train_accuracy.result() * 100, self.test_loss.result(),
                                 self.test_accuracy.result() * 100))
        
        final_accuracy = float(round(self.test_accuracy.result().numpy() * 100,4))

        self.test_loss.reset_state()
        self.train_accuracy.reset_state()
        self.test_accuracy.reset_state()

        stream = io.StringIO()
        self.model.summary(print_fn=lambda x: stream.write(x + '\n'))
        summary_string = stream.getvalue()
        stream.close()
          
        self.save_model_data(summary_string,final_accuracy)
        #final_accuracy = self.test_accuracy.result().numpy()
        return final_accuracy
#    @tf.function
#    def distributed_train_step(dataset_inputs):
#        per_replica_losses = strategy.run(train_step, args=(dataset_inputs,))
#        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)
#    
#    @tf.function
#    def distributed_test_step(dataset_inputs):
#        return strategy.run(test_step, args=(dataset_inputs,))
#    
#    # Inside train_step and test_step, unpack the dataset_inputs:
#    def train_step(dataset_inputs):
#        images, labels = dataset_inputs
#        # Rest of the train_step logic
#        with tf.GradientTape() as tape:
#            predictions = model(images, training=True)
#            loss = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)(labels, predictions)
#        gradients = tape.gradient(loss, model.trainable_variables)
#        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#        return loss
#
#    def test_step(dataset_inputs):
#        images, labels = dataset_inputs
#        # Rest of the test_step logic
#        predictions = model(images, training=False)
#        self.accuracy_metric.update_state(labels, predictions) 
#    
#    def eval_genome_fitness(self, genome) -> float:
#        # TO BE OVERRIDEN
#        raise RuntimeError()
#
#
#
#    def _eval_genome_fitness_weight_training(self, genome):
#        try:
#            with strategy.scope():
#                model = genome.get_model()
#                model.compile(optimizer=Adam(learning_rate=0.0001),
#                              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
#                              metrics=[self.accuracy_metric])
#    
#            for epoch in range(self.epochs):
#                for dataset_inputs in self.train_dataset:
#                    self.distributed_train_step(dataset_inputs)
#            
#            for dataset_inputs in self.test_dataset:
#                self.distributed_test_step(dataset_inputs)
#    
#            final_accuracy = self.accuracy_metric.result().numpy()
#            return float(round(final_accuracy * 100, 4))
#
#    def _eval_genome_fitness_weight_training(self, genome) -> float:
#        try:
#            #strategy = tf.distribute.MirroredStrategy()
#            with strategy.scope():
#                model = genome.get_model()                                                                                                                                    
#                model.summary()
#                #optimizer = genome.get_optimizer()                                                                                                                          
#                optimizer = Adam(learning_rate = 0.0001)
#                #optimizer = mixed_precision.LossScaleOptimizer(optimizer, dynamic=True)
#                print('model compiling,...........')
#                model.compile(optimizer=optimizer,                                                                                                                          
#                              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),                                                                      
#                              metrics=[self.accuracy_metric])                                                                                                                   
#
#            def train_step(images, labels):
#                with tf.GradientTape() as tape:
#                    predictions = model(images, training=True)
#                    loss = tf.keras.losses.BinaryCrossentropy(from_logits=False,reduction=tf.keras.losses.Reduction.SUM)(labels, predictions)
#
#                gradients = tape.gradient(loss, model.trainable_variables)
#                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
#                return loss
#
#            print('Starting model training,...............')
#            #model.fit(self.train_dataset,
#            #                    epochs=self.epochs,
#            #                    verbose=2,
#            #                    steps_per_epoch=1000)
#            #except Exception as e:
#            #    print(f"An error occurred during training: {e}")
#            #    return -1
#
#            for epoch in range(self.epochs):
#                # Iterate over the dataset
#                for images, labels in self.train_dataset:
#                    per_replica_losses = strategy.run(train_step, args=(images, labels))
#                    #reduced_loss = strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None) 
#
#            print('Model training complete,...............')
#            #predictions = model.predict(self.test_dataset)
#            self.accuracy_metric.reset_state()
#            # Check for NaNs in predictions
#                       # Determine fitness by updating the accuracy metric
#            # Iterate over the test dataset batch by batch
#            #i = 0
#            #for image_batch, label_batch in self.test_dataset:
#            #    predictions = model.predict_on_batch(image_batch)
#            #    self.accuracy_metric.update_state(label_batch, predictions)
#           
#                # Iterate over the test dataset in batches and update metric states
#            #for batch_images, batch_labels in self.test_dataset:
#            #    predictions = model(batch_images, training=False)
#            #    self.accuracy_metric.update_state(batch_labels, predictions)
#
#            # After iterating over the test dataset and updating metric states
#            for images, labels in self.test_dataset:
#                predictions = model(images, training=False)
#                self.accuracy_metric.update_state(labels, predictions)
#
#            final_accuracy = self.accuracy_metric.result().numpy()
#            
#            #gc.collect()
#            # Return the final_accuracy, rounded and multiplied by 100
#            return float(round(final_accuracy * 100, 4))
        
#        except Exception as e:
#            # Handle exceptions that occurred during training or evaluation
#            print(f"An exception occurred during training or evaluation: {e}")
            
#            return -1.0  # Set a low fitness score
        
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

