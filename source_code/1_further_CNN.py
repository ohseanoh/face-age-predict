#!/usr/bin/env python
# coding: utf-8

# In[1]:


# %pip install pillow
# %pip install --upgrade pip
# %pip install tensorflow
# %pip install keras
# %pip install --upgrade tensorflow keras
# %pip install opencv-python
# %pip install pydot
# %pip install graphviz
# %pip install keras_vggface
# %pip install virtualenv
# %pip install keras_applications


# In[ ]:


# %pip install tensorflow-gpu


# In[12]:


import json
import shutil
from shutil import copy2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import collections


# In[11]:


from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.utils import plot_model
from sklearn.metrics import classification_report, confusion_matrix


# In[1]:


import tensorflow.keras.layers as Layers
import tensorflow.keras.activations as Actications
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import tensorflow.keras.optimizers as Optimizer
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.metrics as Metrics
import tensorflow.keras.utils as Utils
from tensorflow.keras.utils import model_to_dot
# from keras.utils.vis_utils import model_to_dot
import os
import matplotlib.pyplot as plot
import cv2
import numpy as np
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix as CM
from random import randint
from IPython.display import SVG
import matplotlib.gridspec as gridspec
from keras.preprocessing import image

import tensorflow as tf
from tensorflow import keras


# In[7]:


os.environ["CUDA_VISIBLE_DEVICES"]="0"
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError as e:
        print(e)


# In[8]:


from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# In[9]:


# TensorFlow를 사용하는 경우
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


# In[10]:


import tensorflow as tf

# 사용 가능한 GPU 디바이스 리스트 가져오기
gpu_devices = tf.config.list_physical_devices('GPU')

if gpu_devices:
    print("사용 가능한 GPU가 있습니다.")
else:
    print("사용 가능한 GPU가 없습니다.")


# In[ ]:


for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
    print(f"디바이스 이름: {device.name}")
    print(f"디바이스 유형: {device.device_type}")


# In[4]:


def load_latest_model(directory):
    # 디렉토리 내의 모든 파일 리스트 가져오기
    all_files = os.listdir(directory)

    # .h5 확장자를 가진 파일들만 선택
    model_files = [f for f in all_files if f.endswith(".h5")]

    if not model_files:
        print("No model files found in the directory.")
        return None

    # 파일들의 수정 시간을 기준으로 정렬
    model_files.sort(key=lambda x: os.path.getmtime(os.path.join(directory, x)), reverse=True)

    # 가장 최근에 수정된 모델 파일 불러오기
    latest_model_path = os.path.join(directory, model_files[0])
    
    try:
        loaded_model = load_model(latest_model_path)
        print(f"Loaded the latest model: {latest_model_path}")
        return loaded_model
    except Exception as e:
        print(f"Failed to load the latest model. Error: {e}")
        return None

from datetime import datetime

def save_model(model):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"basic_CNN_{timestamp}.h5"
    model_path = os.path.join('./models_and_weights/', model_filename)
    model.save(model_path)
    return model_filename

def load_model_with_filename(filename):
    if os.path.exists(filename):
        return load_model(filename)
    else:
        print(f"Model file {filename} does not exist.")
        return None


# In[5]:


# model = load_model("./models_and_weights/base_full_model1.h5")
# 특정 디렉토리에서 최근에 저장된 모델 불러오기

learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)

directory_path = "./models_and_weights"
model = load_latest_model(directory_path)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# In[9]:


# dimensions of the images
img_width, img_height = 224, 224

train_data_dir = './classification_data/Training'
test_data_dir = './classification_data/Validation'

# batch size 크면 learning rate도 크게 설정해보기
epochs = 100
batch_size = 256


# In[13]:


# train_datagen = ImageDataGenerator(
#     rescale=1./255,
#     rotation_range=40,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     fill_mode='nearest'
# )
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        directory=train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle = True,
        color_mode = 'rgb')

test_generator = test_datagen.flow_from_directory(
        directory=test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle = False,
        color_mode = 'rgb')


# In[1]:


# history = model.fit_generator(train_generator, steps_per_epoch= 13800//batch_size, epochs=epochs)
history = model.fit_generator(train_generator, epochs=epochs)


# In[ ]:


# 모델 저장
saved_model_filename = save_model(model)
print(f"Model saved as: {saved_model_filename}")


# In[25]:


get_ipython().run_cell_magic('time', '', 'test_generator.reset()\npred=model.predict_generator(test_generator, np.ceil(5050/batch_size))\n')


# In[ ]:


def save_results(predictions, filenames, output_directory):
    # 결과를 저장할 디렉토리 생성
    # os.makedirs(output_directory, exist_ok=True)

    # 현재 시간을 이용한 고유한 파일명 생성
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"results_{timestamp}.csv"

    # 결과 데이터프레임 생성
    results = pd.DataFrame({"Age": filenames, "Predictions": predictions})

    # 결과를 CSV 파일로 저장
    output_path = os.path.join(output_directory, output_filename)
    results.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    return results


# In[26]:


labels = (test_generator.class_indices)
labels = dict((v,k) for k,v in labels.items())
predictions = [labels[k] for k in np.argmax(pred, axis=1)]

filenames=test_generator.filenames
classes = [x.split('\\')[0] for x in filenames]

# results=pd.DataFrame({"Age":classes, "Predictions":predictions})
# results.to_csv('./results/basic_cnn_model_aug.csv',index=False)

# 예측 결과 저장
results = save_results(predictions, classes, "./results")


# In[27]:


results


# In[28]:


# evaluate the model
loss, acc = model.evaluate_generator(train_generator, steps=40150//batch_size)
print('Cross-entropy: ', loss)
print('Accuracy: ', acc)


# In[29]:


# evaluate the model
loss, acc = model.evaluate_generator(test_generator, steps=5050//batch_size)
print('Cross-entropy: ', loss)
print('Accuracy: ', acc)


# In[3]:


plt.plot(history.history['accuracy'])
plt.title('Base Model Accuracy over 100 Epochs')
plt.savefig('./images/base_model_aug_acc.png', bbox_inches='tight')

plt.plot(history.history['loss'])
plt.title('Base Model Loss over 100 Epochs')
plt.savefig('./images/base_model_aug_loss.png', bbox_inches='tight')

