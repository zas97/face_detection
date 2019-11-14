# coding: utf-8

# # Increasing precision
# ## data augmentation
# - flip
# - move
# - blur
# - brighten/darken
# 
# ## network
# hyperparam opt - nb Couches conv, taille filtres
# 
# ## decrease false positive
# increase weight of negatives

# In[28]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix


# In[29]:


home = "C:/Users/JB/Desktop/INSA/5IF/ML/face_detection/dataset/start_deep/"


# ## Loading data

# In[30]:


df = pd.read_csv("dataset/start_deep/posneg.txt", sep=" ", names=["filename", "class"])
df = df.astype(str)
df.head()


# In[31]:


df_test = pd.read_csv("dataset/start_deep/testposneg.txt", sep=" ", names=["filename", "class"])
df_test = df_test.astype(str)
df_test.head()


# In[32]:


# Changing files from pgm to jpg - run only once
# from PIL import Image
# for file in df_test["filename"]:
#     abs_path = home+file
#     img = Image.open(abs_path)
#     new_name = abs_path.replace(".pgm", ".jpg")
#     if new_name != abs_path:
#         img.save(new_name)


# In[33]:


# from PIL import Image
# for file in df["filename"]:
#     abs_path = home+"train_images/" + file
#     img = Image.open(home+"train_images/" + file)
#     new_name = abs_path.replace(".pgm", ".jpg")
#     img.save(new_name)


# In[34]:


# renaming .pgm to jpg
def replace_name(name):
    return name.replace(".pgm", ".jpg")
df["filename"] = df["filename"].apply(replace_name)
df_test["filename"] = df_test["filename"].apply(replace_name)


# In[58]:


df.head()


# ### Split train and validation

# In[36]:


from sklearn.model_selection import train_test_split

df_train, df_val = train_test_split(df)


# In[37]:


root_dir = home+"train_images/"


# In[38]:


from PIL import Image
img = Image.open(root_dir+df["filename"][0])
IMG_HEIGHT, IMG_WIDTH = img.height, img.width
IMG_HEIGHT, IMG_WIDTH


# ### Creating generators

# In[94]:


df.shape
df_train


# In[96]:


batch_size=32
train_image_generator = ImageDataGenerator(rescale=1./255,
#                                                   vertical_flip=True,
#                                                     zca_whitening=True,
                                                          width_shift_range=[-4,4],
                                                          height_shift_range=[-4,4],
                                                          horizontal_flip=True,
                                                          rotation_range=5,
                                                          shear_range=5, # degrees
                                                          zoom_range=0.1, # 20 %
                                                          brightness_range=[0.8,1.0]
                                            )


# generator_to_numpy = ImageDataGenerator().flow_from_dataframe(df_train,
#                                                            directory=root_dir,
#                                                           color_mode="grayscale", x_col="filename", y_col="class",
#                                                           target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=len(df_train),
#                                                           class_mode="binary",
#                                                             )
#
# X, y = generator_to_numpy.next()
# train_image_generator.fit(X)

train_data_gen = train_image_generator.flow_from_dataframe(df_train, 
                                                           directory=root_dir,
                                                          color_mode="grayscale", x_col="filename", y_col="class",
                                                          target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=batch_size,
                                                          class_mode="binary",
                                                          shuffle=True)


# In[97]:


batch_size=32
val_image_generator = ImageDataGenerator(rescale=1./255)
val_data_gen = val_image_generator.flow_from_dataframe(df_val, 
                                                           directory=root_dir,
                                                          color_mode="grayscale", x_col="filename", y_col="class",
                                                          target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=batch_size,
                                                          class_mode="binary",
                                                          shuffle=False)


# In[98]:


test_generator = ImageDataGenerator(rescale=1./255)
test_data_gen = test_generator.flow_from_dataframe(df_test, 
                                                           directory=home,
                                                          color_mode="grayscale", x_col="filename", y_col="class",
                                                          target_size=(IMG_HEIGHT, IMG_WIDTH), batch_size=batch_size,
                                                          class_mode="binary",
                                                          shuffle=True)


# In[99]:


sample_training_images, _ = next(train_data_gen)


# In[100]:


def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img[:,:,0], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


# In[ ]:


plotImages(sample_training_images[:10])


# In[102]:


dropout_rate = 0.3
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,1)),
    MaxPooling2D(),
    Dropout(dropout_rate),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Dropout(dropout_rate),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(dropout_rate),
    Dense(1, activation='sigmoid')
])


# In[103]:


from sklearn.metrics import roc_auc_score

def auc(y_true, y_pred):
    auc = tf.metrics.AUC(y_true, y_pred)[1]
    K.get_session().run(tf.local_variables_initializer())
    return auc

model.compile(optimizer='adam',
              loss="binary_crossentropy", 
             metrics=["accuracy","AUC"])


# In[104]:


model.summary()


# In[105]:


epochs = 5
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=None,
    epochs=epochs,
    validation_data=val_data_gen,
#     validation_steps=len(df) // batch_size
)


# In[106]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# ## Metrics

# In[107]:


truth = val_data_gen.labels
prediction = model.predict_generator(val_data_gen)


# In[108]:


from sklearn.metrics import roc_auc_score, accuracy_score

print(roc_auc_score(truth, prediction))
print(accuracy_score(truth, np.round(prediction)))


# In[109]:


next_batch = next(val_data_gen)
roc_auc_score(next_batch[1], model.predict(next_batch))


# In[110]:


model.evaluate_generator(test_data_gen)


# # Results
# 
# Best enriched data result: [0.0422026319823824, 0.98833245, 0.9972152]
# 
# Without ZCA: [0.08668688679636148, 0.9605401, 0.998305]
# 
# With ZCA:
# 

# In[111]:


# Confusion Matrix and Classification Report

Y_pred = model.predict_generator(val_data_gen, num_of_test_samples // batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))

