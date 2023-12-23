#!/usr/bin/env python
# coding: utf-8

# # Classifying Pictures into Age Groups
# This notebook is used to classify pictures of faces into age groups, based on the given labels. The `shutil` library was used to move pictures from the MegaAge and MegaAge Asian datasets into the appropriate classification folders, in order to match Keras' `.flow_from_directory()` functionality. 
# 
# Before running this notebook, please download all the image data found in the Readme. Furthermore, expect the process of running this notebook completely to take some time.

# In[1]:


import json
import shutil
from shutil import copy2
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import collections


# ## 4-1.
# Training과 Validation dataset에서 파일명 추출해서 모두 한 개의 text file에 저장

# In[4]:


def save_png_filenames_to_txt(root_folder, output_txt_path):
    with open(output_txt_path, 'w') as txt_file:
        # root_folder 밑의 모든 폴더와 파일에 대해 재귀적으로 탐색
        for foldername, subfolders, filenames in os.walk(root_folder):
            for filename in filenames:
                # 파일이 .png로 끝나는 경우에만 파일명을 텍스트 파일에 추가
                if filename.lower().endswith('.png'):
                    # full_path = os.path.join(foldername, filename)
                    full_path = os.path.join(filename)
                    txt_file.write(f"{full_path}\n")

def general_path(train_or_test):
    root = f"./118/{train_or_test}/all_image_unzipped"
    output = f"./text/{train_or_test}_filenames.txt"
    return root, output

# 파일 저장
save_png_filenames_to_txt(*general_path("Training"))
save_png_filenames_to_txt(*general_path("Validation"))

#save_png_filenames_to_txt(*general_path("final_test"))


# ## 4-2.
# Training_filename과 Validation_filename에서 나이 추출해서 모두 한 개의 text file에 저장

# In[6]:


def extract_and_save_words(input_file_path, output_file_path, target_word_index=2):
    with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
        # 파일의 각 줄을 읽어오기
        for line in input_file:
            # 줄을 공백 또는 다른 구분자로 나누기
            words = line.strip().split('_')

            # 특정 단어 추출 및 새로운 파일에 쓰기
            if len(words) > target_word_index:
                target_word = words[target_word_index]
                output_file.write(f"{target_word}\n")

def general_txt(train_or_test):
    root = f"./text/{train_or_test}_filenames.txt"
    output = f"./text/{train_or_test}_age.txt"
    return root, output

# 파일 저장
extract_and_save_words(*general_txt('Training'))
extract_and_save_words(*general_txt('Validation'))

# extract_and_save_words(*general_txt('final_test'))