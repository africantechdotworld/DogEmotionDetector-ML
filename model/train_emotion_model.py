import tensorflow as tf
import os
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

class DogEmotionClassifier:
    def __init__(self, dataset_path, num_classes, img_height=128, img_width=128, batch_size=32, epochs=10):
        self.dataset_path = dataset_path
        self.num_classes = num_classes
        self.img_height, self.img_width = img_height, img_width
        self.batch_size = batch_size
        self.epochs = epochs
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_height, self.img_width, 3)))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D((2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation='relu'))
        model.add(layers.Dense(num_classes, activation='softmax'))  # Assuming multiple classes

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train_model(self):
        # Data preprocessing
        datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

        # Generate training dataset
        train_generator = datagen.flow_from_directory(
            self.dataset_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='training'
        )

        # Generate validation dataset
        validation_generator = datagen.flow_from_directory(
            self.dataset_path,
            target_size=(self.img_height, self.img_width),
            batch_size=self.batch_size,
            class_mode='categorical',
            subset='validation'
        )

        # Train the model
        self.model.fit(train_generator, epochs=self.epochs, validation_data=validation_generator)

    def save_model(self, model_filename):
        self.model.save(model_filename)

# Example usage:
emotion_dataset_dir = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'emotion')
model_filename = "dog_emotion_model.h5"
num_classes = 4  # Replace with the actual number of classes in your dataset

# Create and train the model
dog_emotion_classifier = DogEmotionClassifier(emotion_dataset_dir, num_classes=num_classes)
dog_emotion_classifier.train_model()

# Save the trained model
dog_emotion_classifier.save_model(model_filename)
