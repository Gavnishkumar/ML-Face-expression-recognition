import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from model_architecture import create_emotion_model

# 1. Configuration
# We train for 50 "epochs". An epoch is one full pass through the entire dataset.
# If it takes too long, you can stop it earlier (Ctrl+C), but accuracy will be lower.
EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.0001

# 2. Define the Data Generators (The Conveyor Belt)
# rescale=1./255: Pixels are 0-255 (black to white). 
# Neural networks prefer numbers between 0-1. So we divide by 255.
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

# 3. Connect Generators to Folders
print("Loading Training Data...")
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(48, 48),  # Resize images to 48x48
    batch_size=BATCH_SIZE,
    color_mode="grayscale", # Important: We want 1 channel (gray), not 3 (RGB)
    class_mode='categorical' # We have multiple categories (Happy, Sad, etc.)
)

print("Loading Validation Data...")
validation_generator = val_datagen.flow_from_directory(
    'data/test',
    target_size=(48, 48),
    batch_size=BATCH_SIZE,
    color_mode="grayscale",
    class_mode='categorical'
)

# 4. Create the Model
emotion_model = create_emotion_model()

# Compile with a lower learning rate for stability
emotion_model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=LEARNING_RATE),
    metrics=['accuracy']
)

# 5. Train!
print("Starting training... (This may take a while)")
emotion_model_info = emotion_model.fit(
    train_generator,
    steps_per_epoch=28709 // BATCH_SIZE, # Total train images / batch size
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=7178 // BATCH_SIZE # Total test images / batch size
)


# 6. Save the trained brain
# This file is what we will use in the final app.
# emotion_model.save_weights('emotion_model_weights.h5')
emotion_model.save_weights('emotion_model.weights.h5')
print("Model saved to emotion_model_weights.h5")