Barry Rerecich
Media Design School
barry.rerecich@gmail.com


Working Demo: https://huggingface.co/spaces/barrymoana/CNN-Cat-Dog-Classifier

Can also test model locally by running app.ipynb (given you have a trained model)

### Importing Required Libraries

In this cell, we import all the necessary libraries for the project. This includes libraries for file handling (`os`), image manipulation (`PIL`), numerical operations (`numpy`, `pandas`), data visualization (`matplotlib`, `plotly`), and deep learning (`tensorflow`). Additionally, we import `train_test_split` from `sklearn` to split our dataset into training and validation sets.

Dataset used: 
https://www.kaggle.com/datasets/shaunthesheep/microsoft-catsvsdogs-dataset

```
import os
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go
from plotly.subplots import make_subplots
```

### Loading and Preprocessing Images

Defining the paths to the directories containing images of cats and dogs. Then load these images into two separate lists, ensuring they are in RGB format. This is the first step in data preparation process.

```
# Paths to the dataset directories
cat_dir = 'PetImages/Cat'
dog_dir = 'PetImages/Dog'

# Load the images ensuring they are in RGB format
cat_images = []
for filename in os.listdir(cat_dir):
    if filename.endswith('.jpg'):
        with Image.open(os.path.join(cat_dir, filename)).convert('RGB') as img:
            cat_images.append(img.copy())

dog_images = []
for filename in os.listdir(dog_dir):
    if filename.endswith('.jpg'):
        with Image.open(os.path.join(dog_dir, filename)).convert('RGB') as img:
            dog_images.append(img.copy())
```

### Resizing Images

All cat and dog images are resized to a uniform dimension of 128x128 pixels. This ensures data consistency, which is crucial for training neural network models.


```
# Resize the images
size = (128, 128)
cat_images = [img.resize(size) for img in cat_images]
dog_images = [img.resize(size) for img in dog_images]
```

### Image Augmentation

Initializing an `ImageDataGenerator` object to apply a series of random transformations to each image. These transformations include rotation, width and height shifts, shear transformation, zooming, and horizontal flipping. A demonstration of image augmentation is shown using a single cat image.

```
# Initialize ImageDataGenerator for augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# For demonstration, augmenting a single image
img_array = np.array(cat_images[0]).reshape((1, 128, 128, 3))
augmented_images = next(datagen.flow(img_array, batch_size=1))

plt.imshow(augmented_images[0].astype('uint8'))
plt.show()
```
![Screenshot 2023-08-28 at 2 48 56 PM](https://github.com/barrymoana/CNN_CatDog/assets/62818864/7061cb66-3007-423d-b4e2-e95e49959d45)


### Array Conversion and Labeling

Converting the list of PIL Image objects for cats and dogs into NumPy arrays and checking for inconsistencies in shape. Then, stacking these arrays vertically to create a single array of shape `(n_samples, 128, 128, 3)`. Images are normalized by dividing by 255. Labels are created where 0 denotes a cat and 1 den

```
# Convert cat_images and dog_images to arrays separately
cat_arrays = [np.array(img) for img in cat_images]
dog_arrays = [np.array(img) for img in dog_images]

# Check for inconsistencies in shape within each list
cat_shapes = {arr.shape for arr in cat_arrays}
dog_shapes = {arr.shape for arr in dog_arrays}

cat_shapes, dog_shapes

# Stack images and normalize
X = np.vstack(cat_arrays + dog_arrays).astype('float32') / 255.0

# Create labels (0 for cats, 1 for dogs)
y = np.array([0]*len(cat_arrays) + [1]*len(dog_arrays))

# Check the shapes of X and y
X.shape, y.shape
```

### Reshaping the Feature Array

Reshaping the feature array `X` to ensure it has the correct dimensions, which should be `(n_samples, 128, 128, 3)`.

```
# Had to implenet this given the wrong structure in the above output
X = X.reshape(-1, 128, 128, 3)
X.shape
```

### Splitting the Data and Saving

Splitting the dataset into training and validation sets using a 80-20 ratio. The random state is set to ensure reproducibility. The splits are then saved to a file for future use.

```
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

#saving data splits
np.savez('train_val_data.npz', X_train=X_train, X_val=X_val, y_train=y_train, y_val=y_val)
```

### Loading Saved Data Splits (Optional)

Function to load the previously saved training and validation data splits. This is an optional step in case the notebook is restarted or data needs to be reused.
```
#func for loading saved data splits if needed
loaded_data = np.load('train_val_data.npz')

 Extract the variables from the loaded data
X_train = loaded_data['X_train']
X_val = loaded_data['X_val']
y_train = loaded_data['y_train']
y_val = loaded_data['y_val']
```

### Defining a Simple CNN Architecture

Constructing a simple Convolutional Neural Network (CNN) with one convolutional layer followed by a max-pooling layer. The network also includes a dense hidden layer with 512 neurons and uses a sigmoid activation function for the output layer to enable binary classification. The model is compiled using the Adam optimizer and binary cross-entropy loss function.

``` Simple CNN Architecture
model_simple = Sequential([
    # First Convolutional Layer
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    
    # Flatten the results to feed into a DNN
    Flatten(),
    
    # 512 neuron hidden layer
    Dense(512, activation='relu'),
    
    # Only 1 output neuron as it's binary classification (cat or dog)
    Dense(1, activation='sigmoid')
])

# Compile the model
model_simple.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of the model architecture
model_simple.summary()
```

### Model Training with Augmented Data

Utilizing the `ImageDataGenerator` to augment the training data on-the-fly and fitting the simple CNN model. The model is trained for 10 epochs, and the validation data is used to evaluate its performance at the end of each epoch. After training, the model is saved for future use.

```
# Use datagen.flow() to generate augmented images from X_train and y_train
train_generator = datagen.flow(X_train, y_train, batch_size=32)

# Fit the model using the augmented images
history = model_simple.fit(train_generator, 
                           epochs=10, 
                           validation_data=(X_val, y_val), 
                           steps_per_epoch=len(X_train) // 32)

#After training, saving the model using the save method:
model_simple.save('model_simple.keras')

#how to load model
#model_simple = load_model('model_simple.keras')
```
### Model Evaluation and Result Storage

Evaluating the simple CNN model on the validation set to obtain the final accuracy metric. Also, the training history is converted into a Pandas DataFrame for easier manipulation and analysis in the future.

```
loss, accuracy = model_simple.evaluate(X_val, y_val)
print(f"Validation Accuracy: {accuracy*100:.2f}%")

history_simple_df = pd.DataFrame(history.history)
history_simple_df = history_simple_df.add_prefix('simple_')
```
157/157 [==============================] - 8s 50ms/step - loss: 0.5408 - accuracy: 0.7396
Validation Accuracy: 73.96%

### Visualization of Model Performance

Plotting the training and validation accuracy, as well as loss metrics, over the epochs. This visual representation aids in understanding how the model is learning and whether it is overfitting or underfitting.

```
# Plotting accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plotting loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
```
![Screenshot 2023-08-28 at 2 52 02 PM](https://github.com/barrymoana/CNN_CatDog/assets/62818864/60ccfbc3-145b-4cb3-95a0-05cf5e5395a3)


For More Complex CNNs, rest of the code implentation, and further explainings. Open cnn.ipynb




























