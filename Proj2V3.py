#reminders: When testing is done:
#1. vgg16.trainable = False #set to true when you're further along, leave false just for quicker builds during testing
#2. tf.config.run_functions_eagerly(False) #set to true when you're further along, leave false just for quicker builds during testing
#3. epochs=1 #set to 5 once the notebook is stable
#4a. r1_train_df = r1_df[r1_df['partition'] == 0].sample(frac=0.2, random_state=SEED) #remove sample so you work with the entire dataset once testing is complete
#4b. r1_val_df   = r1_df[r1_df['partition'] == 2].sample(n=2000, random_state=SEED) #remove sample so you work with the entire dataset once testing is complete
#4c. r1_test_df  = r1_df[r1_df['partition'] == 1].sample(n=2000, random_state=SEED) #remove sample so you work with the entire dataset once testing is complete

# Importing the libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import cv2
import random
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools
import warnings

warnings.filterwarnings('ignore')
%matplotlib inline

SEED = 65534
random.seed(SEED)
tf.random.set_seed(SEED)                       # For reproducability.
np.random.seed(SEED)               # ensures numpy-level reproducibility too

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ReduceLROnPlateau

from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model

tf.config.run_functions_eagerly(False) #set to true when you're further along, leave false just for quicker builds during testing


#*************************************************************************************************************************************
#Starter code: import dataset
#*************************************************************************************************************************************
import kagglehub

# Download latest version
path = kagglehub.dataset_download("jessicali9530/celeba-dataset")

print("Path to dataset files:", path)

#*************************************************************************************************************************************
# starter code: read in datasets
#*************************************************************************************************************************************

kaggle_prefix = "" # Note: If we simply put empty string, then the entire code will run in google colab without any issues.

#  read in the annotations

data_og = pd.read_csv(os.path.join(path, "list_attr_celeba.csv"))

#  read in the landmarks
landmarks = pd.read_csv(os.path.join(path, "list_landmarks_align_celeba.csv"))

# read in the partitions to discover train, validation and test segments
partition = pd.read_csv(os.path.join(path, "list_eval_partition.csv"))

image_dir = os.path.join(path, "img_align_celeba", "img_align_celeba")

# selecting only male and bushy eyebrows features. We will also need image_id for unique identfication of these images
#data = data_og[['image_id', 'Male','Young', 'Mustache','Goatee','No_Beard','5_o_Clock_Shadow']]

#LABEL_COLS = [col for col in data.columns if col != 'image_id'] #defines labels for the columns inside of data, excluding image_id
#print(LABEL_COLS)

# transform targets to range (0,1)
#data['Male'].replace({-1: 0}, inplace=True)
#data['Young'].replace({-1: 0}, inplace=True)
#data['Mustache'].replace({-1: 0}, inplace=True)
#data['Goatee'].replace({-1: 0}, inplace=True)
#data['No_Beard'].replace({-1: 0}, inplace=True)
#data['5_o_Clock_Shadow'].replace({-1: 0}, inplace=True)
#data['Wearing_Lipstick'].replace({-1: 0}, inplace=True)

# perform an inner join of the result with the partition data frame on image_id to obtain integrated partitions
#df = pd.merge(data, partition, on='image_id', how='inner')





#*************************************************************************************************************************************
#starter code: Data generator class
#*************************************************************************************************************************************

# Data generator class - this is a key class that is used to batch the data so as to
# reduce compute time as well as to fit training segments into available memory
# Additionally it allows you to specify multiple targets for classification
# Also allows for image cropping
from tensorflow.keras.preprocessing import image
from tensorflow.python.keras.utils import np_utils

class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, df, label_cols, image_dir, batch_size=32, dim=(218,178), n_channels=3, n_classes=2, shuffle=False):
        self.dim = dim
        self.batch_size = batch_size
        self.df = df
        self.label_cols = label_cols
        self.image_dir = image_dir
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size)) # Modified code here to include the last batch as well.

    def __getitem__(self, index):
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # saves memory by batching
        df_temp = self.df.iloc[indexes].reset_index(drop=True)
        X, y = self.__data_generation(df_temp)
        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, df_temp):
        # Adjust for the last batch which might be smaller
        current_batch_size = len(df_temp)
        X = np.empty((current_batch_size, *self.dim, self.n_channels), dtype=np.float32)
        y = np.empty((current_batch_size, self.n_classes), dtype=np.float32)

        for i, row in df_temp.iterrows():
                img_path = os.path.join(self.image_dir, row["image_id"])
                img = image.load_img(img_path, target_size=self.dim)
                img = image.img_to_array(img)
                img /= 255.0
                X[i] = img
                y[i] = row[self.label_cols].values.astype(np.float32) # need to assign y[i] accordingly to the row object to specify the target(s) for the query.
        return X, y

#testing out male only dataframe
r1_data = data_og[['image_id', 'Male']].copy()
R1_LABEL_COLS = [col for col in r1_data.columns if col != 'image_id'] #defines labels for the columns inside of data, excluding image_id
print(R1_LABEL_COLS)
# transform targets to range (0,1)
r1_data['Male'].replace({-1: 0}, inplace=True)
# perform an inner join of the result with the partition data frame on image_id to obtain integrated partitions
r1_df = pd.merge(r1_data, partition, on='image_id', how='inner')



#*************************************************************************************************************************************
#starter code: split partitions
#*************************************************************************************************************************************

#train_df = df[df['partition'] == 0] #if partition number = 0, that goes into the train df variable
#test_df = df[df['partition'] == 1]
#val_df = df[df['partition'] == 2]

#testing out male only split partitions
r1_train_df = r1_df[r1_df['partition'] == 0].sample(frac=0.2, random_state=SEED) #if partition number = 0, that goes into the train df variable
r1_val_df   = r1_df[r1_df['partition'] == 2].sample(n=2000, random_state=SEED)
r1_test_df  = r1_df[r1_df['partition'] == 1].sample(n=2000, random_state=SEED)


#*************************************************************************************************************************************
#starter code: sample partitions
#*************************************************************************************************************************************

r1_train_df.head()
print (len(r1_train_df))


r1_val_df.head()
print (len(r1_val_df))

print(len(r1_test_df))
print(r1_test_df.head())

print(len(r1_train_df), len(r1_val_df), len(r1_test_df))
#*************************************************************************************************************************************
# starter code: # lengths of dfs
#*************************************************************************************************************************************
print(f"Lengths of train, validation and test partitions: {len(r1_train_df), len(r1_test_df), len(r1_val_df)}")

print(f"Attributes: \n{r1_data.head(10)}\n\n")
print(f"Landmarks: \n{landmarks.head(2)}\n\n")
print(f"partitions: \n{partition.head(2)}\n\n")

print(f"Training Data: \n{r1_train_df.head(2)}\n\n")
print(f"Validation Data: \n{r1_val_df.head(2)}\n\n")
print(f"Test Data: \n{r1_test_df.head(2)}\n\n")
#*************************************************************************************************************************************
# starter code: # using vgg16 as feature extractor
#*************************************************************************************************************************************

vgg16 = tf.keras.applications.VGG16(input_shape=(218, 178, 3), include_top=False, weights='imagenet')
vgg16.trainable = False #set to true when you're further along, leave false just for quicker builds during testing

# creating the model
r1_model = tf.keras.Sequential([
    vgg16,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(len(R1_LABEL_COLS), activation='sigmoid')
])



#*************************************************************************************************************************************
# starter code: compiling the model
#*************************************************************************************************************************************
r1_model.compile(optimizer=SGD(learning_rate=0.0001), loss='binary_crossentropy', metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)])



#*************************************************************************************************************************************
# starter code: creating the train, test and validation data generators
#*************************************************************************************************************************************
r1_train_generator = DataGenerator(r1_train_df,R1_LABEL_COLS, image_dir, batch_size=32, dim=(218,178), n_channels=3,n_classes=len(R1_LABEL_COLS), shuffle=True)
r1_test_generator = DataGenerator(r1_test_df,R1_LABEL_COLS, image_dir, batch_size=32, dim=(218,178), n_channels=3,n_classes=len(R1_LABEL_COLS), shuffle=False)
r1_val_generator = DataGenerator(r1_val_df,R1_LABEL_COLS, image_dir, batch_size=32, dim=(218,178), n_channels=3,n_classes=len(R1_LABEL_COLS), shuffle=False)

training = r1_model.fit(
    r1_train_generator,
    validation_data=r1_val_generator,
    epochs=1
)

#*************************************************************************************************************************************
#starter code: checking test generator
#*************************************************************************************************************************************

import sys
print(r1_test_generator[0][0][0].shape)
print(sys.getsizeof(r1_test_generator[0][0][0]))
print(r1_test_generator[0][0][0])


#*************************************************************************************************************************************
#Starter code: evaluate model
#*************************************************************************************************************************************

print(r1_model.evaluate(r1_test_generator,batch_size=32))

#*************************************************************************************************************************************
#Starter code: predict on unseen data
#*************************************************************************************************************************************

r1_prediction = r1_model.predict(r1_test_generator)
test_array=r1_test_df.to_numpy()
count=0
#print(test_df.head(10))
#print(prediction[:,:])
for pred in r1_prediction:
    if(count<29): count=count+1 #print(pred)

test_size=len(r1_test_df)
correct=0

for i in range(test_size):
    pred_class1 = r1_prediction[i,0]
    #pred_class1 = r1_prediction[i,0]
    test_class1 = test_array[i,1]

    #pred_class2= r1_prediction[i,1]
    #test_class2= test_array[i,2]

    if(np.where(pred_class1 > 0.5, 1,0))==test_class1: # in pred class 1, where the value is greater than 0.5, test class = 1, otherwise, test class = 0
       #and (np.where(pred_class2 > 0.5, 1,0))==test_class2): 
      correct=correct+1
    #print(f"{i}, {pred_class1:.2f}, {test_class1}, {pred_class2:.2f}, {test_class2}")
    if i < 20:
      print(f"{i}, {pred_class1:.2f}, {test_class1}")
print(f"test accuracy is: {(correct/test_size):.1%}")


#*************************************************************************************************************************************
#R1 - code to classify values as 1 or 0 easier
#*************************************************************************************************************************************

def classify_value(value, threshold):
    if value > threshold:
        return 1
    else:
        return 0

#test accuracy for male class only
correct = 0
for i in range(test_size):
    pred_class1 = r1_prediction[i,0]
    test_class1 = test_array[i,1]
    #print(f"{i}, {pred_class1:.2f}, {test_class1}, {pred_class2:.2f}, {test_class2}")
    if((np.where(pred_class1 > 0.5, 1,0))==test_class1): correct=correct+1
print(f"male only test accuracy is: {(correct/test_size):.1%}")


#another way of testing male accuracy
r1_test_df_reset = r1_test_df.reset_index(drop=True)
y_true = r1_test_df_reset[R1_LABEL_COLS].to_numpy().astype(int)
y_pred = (r1_prediction[:len(y_true)] > 0.5).astype(int)

for j, col in enumerate(R1_LABEL_COLS):
    acc = (y_true[:, j] == y_pred[:, j]).mean()
    print(f"{col:18s}: {acc:.1%}")


#*************************************************************************************************************************************
#R1 - Added some code to create a confusion matrix of the predictions
#*************************************************************************************************************************************

#confusion matrix currently only shows accuracy for one class = male


from sklearn import metrics
import matplotlib.pyplot as plt
#for i in prediction:

r1_pclassified = np.zeros([test_size])
for i in range(test_size):
  r1_pclassified[i] = classify_value(r1_prediction[i,0], 0.5)

r1_pclassified = r1_pclassified.astype(int)

print(len(r1_pclassified))

r1_tclassified = np.zeros([test_size])
for i in range(test_size):
  r1_tclassified[i] = test_array[i,1]

r1_cm = metrics.confusion_matrix(r1_tclassified, r1_pclassified)

cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=r1_cm, display_labels=['Female', 'Male'])

cm_display.plot()
plt.show()


#confusion matrix accuracy reading
TN = r1_cm[0,0]
FP = r1_cm[0,1]
FN = r1_cm[1,0]
TP = r1_cm[1,1]

accuracy = ((TP + TN) / r1_cm.sum())
print(f"Classification accuracy for male or female: {(accuracy):.1%}")

Xb, yb = r1_train_generator[0]
print("Xb:", Xb.shape, Xb.dtype, Xb.min(), Xb.max())
print("yb:", yb.shape, yb.dtype)
print("First 10 y rows:\n", yb[:10])
print("Unique values in y:", np.unique(yb))

#*************************************************************************************************************************************
# R2 - Accuracy based on age
#*************************************************************************************************************************************

from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import sys

CLASS_NAMES = ['Old-Woman', 'Young-Woman', 'Old-Man', 'Young-Man']

#*************************************************************************************************************************************
# R2 - helper function to convert Male + Young into 4 classes
#*************************************************************************************************************************************

def convert_to_4class(male_value, young_value):
    # male_value: 0 = female, 1 = male
    # young_value: 0 = old, 1 = young
    if male_value == 0 and young_value == 0:
        return 0   # Old-Woman
    elif male_value == 0 and young_value == 1:
        return 1   # Young-Woman
    elif male_value == 1 and young_value == 0:
        return 2   # Old-Man
    else:
        return 3   # Young-Man


#*************************************************************************************************************************************
# R2 - create dataframe with Male and Young
#*************************************************************************************************************************************

r2_data = data_og[['image_id', 'Male', 'Young']].copy()
r2_LABEL_COLS = [col for col in r2_data.columns if col != 'image_id']
print(r2_LABEL_COLS)

r2_data['Male'].replace({-1: 0}, inplace=True)
r2_data['Young'].replace({-1: 0}, inplace=True)

r2_df = pd.merge(r2_data, partition, on='image_id', how='inner')


#*************************************************************************************************************************************
# R2 - split partitions
#*************************************************************************************************************************************

r2_train_df = r2_df[r2_df['partition'] == 0].sample(frac=0.2, random_state=SEED)
r2_val_df   = r2_df[r2_df['partition'] == 2].sample(n=2000, random_state=SEED)
r2_test_df  = r2_df[r2_df['partition'] == 1].sample(n=2000, random_state=SEED)

print(len(r2_train_df))
print(len(r2_val_df))
print(len(r2_test_df))
print(r2_test_df.head())

print(f"Lengths of train, validation and test partitions: {len(r2_train_df), len(r2_val_df), len(r2_test_df)}")

print(f"Attributes: \n{r2_data.head(10)}\n\n")
print(f"Landmarks: \n{landmarks.head(2)}\n\n")
print(f"partitions: \n{partition.head(2)}\n\n")

print(f"Training Data: \n{r2_train_df.head(2)}\n\n")
print(f"Validation Data: \n{r2_val_df.head(2)}\n\n")
print(f"Test Data: \n{r2_test_df.head(2)}\n\n")


#*************************************************************************************************************************************
# R2 Part 1 - M1
# Use the R1 model and evaluate whether it generalizes with age
#*************************************************************************************************************************************

print("====================================================")
print("R2 Part 1 - M1: Evaluate R1 gender model on 4 classes")
print("====================================================")

r2_m1_test_df = r2_test_df[['image_id', 'Male']].copy()

r2_m1_test_generator = DataGenerator(
    r2_m1_test_df,
    ['Male'],
    image_dir,
    batch_size=32,
    dim=(218,178),
    n_channels=3,
    n_classes=1,
    shuffle=False
)

print(r1_model.evaluate(r2_m1_test_generator, batch_size=32))

m1_prediction = r1_model.predict(r2_m1_test_generator)
r2_test_df_reset = r2_test_df.reset_index(drop=True)

m1_true_4class = np.zeros(len(r2_test_df_reset), dtype=int)
m1_pred_4class = np.zeros(len(r2_test_df_reset), dtype=int)

for i in range(len(r2_test_df_reset)):
    pred_male = int(m1_prediction[i,0] >= 0.5)
    true_male = int(r2_test_df_reset.loc[i, 'Male'])
    true_young = int(r2_test_df_reset.loc[i, 'Young'])

    m1_true_4class[i] = convert_to_4class(true_male, true_young)
    m1_pred_4class[i] = convert_to_4class(pred_male, true_young)

    if i < 20:
        print(f"{i}, pred_male={pred_male}, true_male={true_male}, true_young={true_young}")

m1_accuracy = (m1_true_4class == m1_pred_4class).mean()
print(f"M1 4-class test accuracy is: {m1_accuracy:.1%}")

m1_cm = metrics.confusion_matrix(m1_true_4class, m1_pred_4class)

cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=m1_cm,
    display_labels=CLASS_NAMES
)
cm_display.plot(cmap='viridis')
plt.title("R2 Part 1 - M1 confusion matrix")
plt.xticks(rotation=45)
plt.show()

print("Classification report for M1:")
print(metrics.classification_report(m1_true_4class, m1_pred_4class, target_names=CLASS_NAMES))


#*************************************************************************************************************************************
# R2 Part 2 - M2
# Train a new model using both Male and Young as targets
#*************************************************************************************************************************************

print("====================================================")
print("R2 Part 2 - M2: Train model with Male and Young targets")
print("====================================================")

vgg16 = tf.keras.applications.VGG16(input_shape=(218, 178, 3), include_top=False, weights='imagenet')
vgg16.trainable = False

r2_model = tf.keras.Sequential([
    vgg16,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(1024, activation='relu'),
    tf.keras.layers.Dense(len(r2_LABEL_COLS), activation='sigmoid')
])

r2_model.compile(
    optimizer=SGD(learning_rate=0.0001),
    loss='binary_crossentropy',
    metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0.5)]
)

r2_train_generator = DataGenerator(
    r2_train_df, r2_LABEL_COLS, image_dir,
    batch_size=32, dim=(218,178), n_channels=3,
    n_classes=len(r2_LABEL_COLS), shuffle=True
)

r2_val_generator = DataGenerator(
    r2_val_df, r2_LABEL_COLS, image_dir,
    batch_size=32, dim=(218,178), n_channels=3,
    n_classes=len(r2_LABEL_COLS), shuffle=False
)

r2_test_generator = DataGenerator(
    r2_test_df, r2_LABEL_COLS, image_dir,
    batch_size=32, dim=(218,178), n_channels=3,
    n_classes=len(r2_LABEL_COLS), shuffle=False
)

r2_training = r2_model.fit(
    r2_train_generator,
    validation_data=r2_val_generator,
    epochs=1
)

#*************************************************************************************************************************************
# R2 - check test generator
#*************************************************************************************************************************************

print(r2_test_generator[0][0][0].shape)
print(sys.getsizeof(r2_test_generator[0][0][0]))
print(r2_test_generator[0][0][0])

#*************************************************************************************************************************************
# R2 - evaluate model
#*************************************************************************************************************************************

print(r2_model.evaluate(r2_test_generator, batch_size=32))

#*************************************************************************************************************************************
# R2 - predict on unseen data
#*************************************************************************************************************************************

r2_prediction = r2_model.predict(r2_test_generator)
test_size = len(r2_test_df)

for i in range(min(5, len(r2_prediction))):
    print(r2_prediction[i])

#*************************************************************************************************************************************
# R2 - accuracy for each separate target
#*************************************************************************************************************************************

r2_test_df_reset = r2_test_df.reset_index(drop=True)
y_true = r2_test_df_reset[r2_LABEL_COLS].to_numpy().astype(int)
y_pred = (r2_prediction[:len(y_true)] >= 0.5).astype(int)

for j, col in enumerate(r2_LABEL_COLS):
    acc = (y_true[:, j] == y_pred[:, j]).mean()
    print(f"{col:18s}: {acc:.1%}")

#*************************************************************************************************************************************
# R2 - overall 4-class accuracy
#*************************************************************************************************************************************

r2_true_4class = np.zeros(test_size, dtype=int)
r2_pred_4class = np.zeros(test_size, dtype=int)

for i in range(test_size):
    pred_male = int(r2_prediction[i,0] >= 0.5)
    pred_young = int(r2_prediction[i,1] >= 0.5)

    true_male = int(r2_test_df_reset.loc[i, 'Male'])
    true_young = int(r2_test_df_reset.loc[i, 'Young'])

    r2_true_4class[i] = convert_to_4class(true_male, true_young)
    r2_pred_4class[i] = convert_to_4class(pred_male, pred_young)

    if i < 20:
        print(f"{i}, pred_male={pred_male}, true_male={true_male}, pred_young={pred_young}, true_young={true_young}")

m2_accuracy = (r2_true_4class == r2_pred_4class).mean()
print(f"M2 4-class test accuracy is: {m2_accuracy:.1%}")

#*************************************************************************************************************************************
# R2 - confusion matrix for 4 classes
#*************************************************************************************************************************************

r2_cm = metrics.confusion_matrix(r2_true_4class, r2_pred_4class)

cm_display = metrics.ConfusionMatrixDisplay(
    confusion_matrix=r2_cm,
    display_labels=CLASS_NAMES
)
cm_display.plot(cmap='viridis')
plt.title("R2 Part 2 - M2 confusion matrix")
plt.xticks(rotation=45)
plt.show()

print("Classification report for M2:")
print(metrics.classification_report(r2_true_4class, r2_pred_4class, target_names=CLASS_NAMES))


#*************************************************************************************************************************************
# R2 - print final comparison
#*************************************************************************************************************************************

print("====================================================")
print("R2 Final Comparison")
print("====================================================")
print(f"M1 4-class accuracy: {m1_accuracy:.1%}")
print(f"M2 4-class accuracy: {m2_accuracy:.1%}")

if m2_accuracy > m1_accuracy:
    print("M2 performed better than M1.")
    print("This suggests that gender classification does not generalize well with age.")
else:
    print("M2 did not perform better than M1.")
    print("This suggests that gender classification may generalize with age in this experiment.")


#*************************************************************************************************************************************
# R2 - inspect one batch
#*************************************************************************************************************************************

Xb, yb = r2_train_generator[0]
print("Xb:", Xb.shape, Xb.dtype, Xb.min(), Xb.max())
print("yb:", yb.shape, yb.dtype)
print("First 10 y rows:\n", yb[:10])
print("Unique values in y:", np.unique(yb))
