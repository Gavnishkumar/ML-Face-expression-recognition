import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def create_emotion_model():
    # We use a "Sequential" model, which means we stack layers like a sandwich.
    model = Sequential()

    # --- LAYER 1: The Input Layer ---
    # Conv2D: This layer creates 32 "filters" (little scanners).
    # kernel_size=(3,3): Each scanner is a 3x3 pixel grid.
    # activation='relu': This is a math function that turns negative numbers to 0. 
    # It helps the AI learn non-linear patterns (like curves).
    # input_shape=(48, 48, 1): We expect a 48x48 pixel image with 1 color channel (Grayscale).
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
    
    # --- LAYER 2: The Downsizing Layer ---
    # MaxPooling2D: This takes the largest value from a 2x2 grid. 
    # Effectively, it shrinks the image by half but keeps the most important features.
    # This reduces calculation time and prevents "overfitting" (memorizing the image).
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Dropout: Randomly turns off 25% of neurons during training.
    # This forces the brain to not rely on just one neuron—it makes it robust.
    model.add(Dropout(0.25))

    # --- LAYER 3 & 4: Deepening the Network ---
    # We add more filters (64) to find more complex patterns (eyes, mouth shapes).
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # --- LAYER 5 & 6: Even Deeper ---
    # 128 filters. Now we are looking for whole facial structures.
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # --- LAYER 7: Flattening ---
    # Up to here, we had 2D grids (images). Now we need to make a decision.
    # Flatten converts the 2D grid into a single long list of numbers (1D array).
    model.add(Flatten())

    # --- LAYER 8: The Thinking Layer ---
    # Dense: A standard neural network layer with 1024 neurons to process the features found above.
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))

    # --- LAYER 9: The Output Layer ---
    # Dense(7): We have 7 neurons, one for each emotion.
    # activation='softmax': This converts the numbers into percentages that add up to 100%.
    # Example output: [0.1, 0.8, 0.0, ...] -> 10% Angry, 80% Happy...
    model.add(Dense(7, activation='softmax'))

    # Compile the model
    # optimizer='adam': The algorithm that adjusts the weights to minimize errors.
    # loss='categorical_crossentropy': How we measure how "wrong" the model is.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

if __name__ == "__main__":
    model = create_emotion_model()
    # This prints a summary of the architecture to your terminal
    model.summary()