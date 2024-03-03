import os
from PIL import Image, ImageFilter
import numpy as np
import threading
import random
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class DataTransformer:
    def transform(folder, dimension, destination_folder,
                  interpolation: bool = True, denoise: bool = True):
        image_files = os.listdir(folder)

        if not os.path.exists(destination_folder):
            os.makedirs(destination_folder)

        threads = []
        for image_file in image_files:
            image_path = os.path.join(folder, image_file)
            thread = threading.Thread(target=DataTransformer._transform_image,
                                      args=(image_path, dimension,
                                            destination_folder,
                                            interpolation, denoise))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    def transform_image(image_path, dimension: int = 64,
                        interpolation: bool = True, denoise: bool = True):
        img = DataTransformer._transform(image_path, dimension,
                                         interpolation, denoise)
        return np.array(img) / 255.0

    def _transform(image_path, dimension, interpolation, denoise):
        '''
         Description: Performs the actual image transformation,
         including resizing, interpolation, and denoising.
         This method is used internally by transform and transform_image
         methods.
        '''
        img = Image.open(image_path)
        if interpolation:
            # Resize and interpolate with Bicubic method
            img = img.resize((dimension, dimension), Image.BICUBIC)
        else:
            img = img.resize((dimension, dimension))

        if img.mode != 'RGB':
            img = img.convert('RGB')
        if denoise:
            # Image denoising
            img = img.filter(ImageFilter.BLUR)
            img = img.filter(ImageFilter.MinFilter(size=3))
            img = img.filter(ImageFilter.MinFilter)
        return img

    def _transform_image(image_path, dimension, destination_folder,
                         interpolation, denoise):
        '''
        Description: Performs image transformation and saves the preprocessed
        image to the specified destination folder.
        This method is used internally by the transform method.
        '''
        img = DataTransformer._transform(image_path, dimension,
                                         interpolation, denoise)
        # Save the denoised image to the destination folder
        destination_path = os.path.join(destination_folder,
                                        os.path.basename(image_path))
        img.save(destination_path)


class DataExtractor:
    def __init__(self, train_benign: str, train_malign: str,
                 test_benign: str, test_malign: str, augment: int = 0):
        self._train_data = []
        self._test_data = []
        self._list_lock = threading.Lock()

        # Type refers to training data (0) or testing data (1)
        self._extract(train_benign, label=0, type=0)
        self._extract(train_malign, 1, 0)
        self._extract(test_benign, 0, 1)
        self._extract(test_malign, 1, 1)

        if augment > 0:
            self._augment(augment)

    def generate_train_test_data(self):
        random.shuffle(self._train_data)

        training_data = np.array(list(map(lambda x: x[0], self._train_data)))
        training_labels = np.array(list(map(lambda x: x[1], self._train_data)))
        testing_data = np.array(list(map(lambda x: x[0], self._test_data)))
        testing_labels = np.array(list(map(lambda x: x[1], self._test_data)))

        return training_data, training_labels, testing_data, testing_labels

    def _extract(self, folder, label, type):
        '''
         Description: Extracts images from a specified folder,
         normalizes them, and assigns labels.
        '''
        image_files = os.listdir(folder)
        threads = []
        for image_file in image_files:
            thread = threading.Thread(target=self._image_extract,
                                      args=(folder, label, type, image_file,))
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()

    def _image_extract(self, folder, label, type, image_file):
        '''
         Description: Method used internally by _extract method
        '''
        image_path = os.path.join(folder, image_file)
        img = Image.open(image_path)

        with self._list_lock:
            if type == 0:
                self._train_data.append((np.array(img) / 255.0, label))
            else:
                self._test_data.append((np.array(img) / 255.0, label))

    def _augment(self, augmentation_factor):
        gen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.2,
            zoom_range=0.2,
            fill_mode='nearest')

        n = len(self._train_data)
        for i in range(n):
            img = self._train_data[i][0]
            label = self._train_data[i][1]
            img_data = np.expand_dims(img, axis=0)
            augmented_images = gen.flow(img_data, batch_size=1)

            for _ in range(augmentation_factor):
                augmented_image = augmented_images.next()[0]
                self._train_data.append((augmented_image, label))

        n = len(self._test_data)
        for i in range(n):
            img = self._test_data[i][0]
            label = self._test_data[i][1]
            img_data = np.expand_dims(img, axis=0)
            augmented_images = gen.flow(img_data, batch_size=1)

            for _ in range(augmentation_factor):
                augmented_image = augmented_images.next()[0]
                self._test_data.append((augmented_image, label))
