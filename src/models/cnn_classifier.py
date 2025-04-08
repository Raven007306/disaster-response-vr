"""
CNN-based classifier for satellite imagery.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import logging

logger = logging.getLogger(__name__)

class DisasterClassifier:
    """
    CNN-based classifier for identifying disaster-affected areas in satellite imagery.
    
    This model can classify satellite imagery into different disaster types
    (flood, wildfire, earthquake damage, etc.) or identify affected vs. unaffected areas.
    """
    
    def __init__(self, input_shape=(256, 256, 3), num_classes=2):
        """
        Initialize the disaster classifier.
        
        Args:
            input_shape (tuple): Shape of input images (height, width, channels).
            num_classes (int): Number of output classes.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self):
        """
        Build the CNN model architecture.
        
        Returns:
            tensorflow.keras.Model: The compiled model.
        """
        try:
            # Create a sequential model
            model = models.Sequential()
            
            # Add convolutional layers
            model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(64, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(128, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            model.add(layers.Conv2D(128, (3, 3), activation='relu'))
            model.add(layers.MaxPooling2D((2, 2)))
            
            # Add dense layers
            model.add(layers.Flatten())
            model.add(layers.Dropout(0.5))
            model.add(layers.Dense(512, activation='relu'))
            model.add(layers.Dense(self.num_classes, activation='softmax'))
            
            # Compile the model
            model.compile(
                optimizer=optimizers.Adam(1e-4),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.model = model
            logger.info(f"Built CNN classifier with {self.num_classes} output classes")
            return model
            
        except Exception as e:
            logger.error(f"Error building model: {e}")
            raise
    
    def train(self, train_data, train_labels, validation_data=None, 
              epochs=20, batch_size=32, callbacks=None):
        """
        Train the model on the provided data.
        
        Args:
            train_data (numpy.ndarray): Training data.
            train_labels (numpy.ndarray): Training labels.
            validation_data (tuple, optional): Validation data and labels.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            callbacks (list, optional): List of Keras callbacks.
            
        Returns:
            History: Training history.
        """
        try:
            if self.model is None:
                self.build_model()
                
            # Create default callbacks if none provided
            if callbacks is None:
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss', patience=5, restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6
                    )
                ]
            
            # Train the model
            history = self.model.fit(
                train_data, train_labels,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=validation_data,
                callbacks=callbacks
            )
            
            logger.info(f"Model trained for {len(history.epoch)} epochs")
            return history
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def predict(self, data):
        """
        Make predictions with the trained model.
        
        Args:
            data (numpy.ndarray): Input data for prediction.
            
        Returns:
            numpy.ndarray: Model predictions.
        """
        try:
            if self.model is None:
                raise ValueError("Model has not been built or trained")
                
            predictions = self.model.predict(data)
            return predictions
            
        except Exception as e:
            logger.error(f"Error making predictions: {e}")
            raise
    
    def save_model(self, model_path):
        """
        Save the trained model to disk.
        
        Args:
            model_path (str): Path to save the model.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            if self.model is None:
                raise ValueError("No model to save")
                
            # Ensure directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save the model
            self.model.save(model_path)
            logger.info(f"Model saved to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, model_path):
        """
        Load a trained model from disk.
        
        Args:
            model_path (str): Path to the saved model.
            
        Returns:
            bool: True if successful, False otherwise.
        """
        try:
            self.model = models.load_model(model_path)
            logger.info(f"Model loaded from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False 