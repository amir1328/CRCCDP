import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# Constants
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR = os.path.join(BASE_DIR, "Train")
MODEL_PATH = os.path.join(BASE_DIR, "model.h5")

def train_model():
    print(f"Checking for training data in: {TRAIN_DIR}")
    if not os.path.exists(TRAIN_DIR):
        print("Train directory not found!")
        return

    # Data Augmentation and Loading
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2 
    )

    print("Loading images...")
    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    validation_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    num_classes = train_generator.num_classes
    class_names = list(train_generator.class_indices.keys())
    print(f"Found {num_classes} classes: {class_names}")

    # Save class names for the app to use later
    with open("class_names.txt", "w") as f:
        f.write("\n".join(class_names))

    # Base Model (Transfer Learning)
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    
    # Freeze base model
    base_model.trainable = False

    # Custom Head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.inputs, outputs=predictions)

    model.compile(optimizer=Adam(learning_rate=0.0001),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    print("Starting training...")
    model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=validation_generator
    )

    print(f"Saving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    print("Training complete!")

if __name__ == "__main__":
    train_model()
