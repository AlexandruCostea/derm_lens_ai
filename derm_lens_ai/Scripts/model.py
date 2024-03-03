import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from data import DataTransformer, DataExtractor


class CNNModel:
    def __init__(self, model: keras.models.Model = None,
                 train_data: np.ndarray = None,
                 train_labels: np.ndarray = None,
                 test_data: np.ndarray = None,
                 test_labels: np.ndarray = None):
        self._label_names = ['Benign', 'Malignant']
        self._train_data = train_data
        self._train_labels = train_labels
        self._test_data = test_data
        self._test_labels = test_labels

        if model is not None:
            self._model = model
        else:
            self.create_model()

    def create_model(self):
        # Convolutional layers
        self._model = keras.models.Sequential()
        self._model.add(keras.layers.Conv2D(64, (3, 3),
                                            activation='relu',
                                            input_shape=(64, 64, 3)))
        self._model.add(keras.layers.MaxPooling2D((2, 2)))
        self._model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
        self._model.add(keras.layers.MaxPooling2D((2, 2)))
        self._model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
        self._model.add(keras.layers.MaxPooling2D((2, 2)))
        self._model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))

        # Dense layers
        self._model.add(keras.layers.Flatten())
        self._model.add(keras.layers.Dense(64, activation='relu'))
        self._model.add(keras.layers.Dense(1, activation='sigmoid'))

        base_learning_rate = 0.0001
        self._model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
                            loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
                            metrics=['accuracy'])

    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model: keras.models.Model):
        self._model = model

    @property
    def train_dataset(self):
        return self._train_data, self._train_labels

    @train_dataset.setter
    def train_dataset(self, train_data: np.ndarray, train_labels: np.ndarray):
        if train_data is None or train_labels is None:
            raise Exception('Train dataset can not contain None elements')
        if len(train_data) != len(train_labels):
            raise Exception('Training data and labels array must be of same length')
        self._train_data = train_data
        self._train_labels = train_labels

    @property
    def test_dataset(self):
        return self._test_data, self._train_labels

    @test_dataset.setter
    def test_dataset(self, test_data: np.ndarray, test_labels: np.ndarray):
        if test_data is None or test_labels is None:
            raise Exception('Test dataset can not contain None elements')
        if len(test_data) != len(test_labels):
            raise Exception('Testing data and labels array must be of same length')
        self._test_data = test_data
        self._test_labels = test_labels

    def summary(self):
        self._model.summary()

    def fit(self, epochs=10, batch_size=128):
        if self._train_data is not None and self._train_labels is not None:
            if len(self._train_data) == len(self._train_labels):
                self._model.fit(self._train_data, self._train_labels,
                                epochs=epochs, batch_size=batch_size)
            else:
                raise Exception('Training data and labels do not match')
        else:
            raise Exception('No training dataset provided')

    def evaluate(self, batch_size=128):
        if self._test_data is not None and self._test_labels is not None:
            if len(self._test_data) == len(self._test_labels):
                loss, accuracy = self._model.evaluate(self._test_data,
                                                      self._test_labels,
                                                      batch_size=batch_size)
                return loss, accuracy
            else:
                raise Exception('Testing data and labels do not match')
        raise Exception('No testing dataset provided')

    def predict(self, dataset: np.ndarray):
        return self._model.predict(dataset)

    def predict_for_image(self, image_path: str,
                          resize_dimension: int = 64,
                          interpolation: bool = True,
                          denoise: bool = True):
        img = DataTransformer.transform_image(image_path,
                                              resize_dimension,
                                              interpolation,
                                              denoise)
        img = np.expand_dims(img, axis=0)
        prediction = self._model.predict(img)[0][0]
        if prediction >= 0.5:
            prediction_label = self._label_names[1]
            confidence = prediction
        else:
            prediction_label = self._label_names[0]
            confidence = 1 - prediction

        return prediction_label, confidence

    def save(self, name: str):
        self._model.save(name)
