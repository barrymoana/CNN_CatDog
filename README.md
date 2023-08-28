Working Demo: https://huggingface.co/spaces/barrymoana/CNN-Cat-Dog-Classifier

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

