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


# ## 1.
# [USE ONLY ONCE] 나이별로 구분되어 있는 classification 폴더 만들기

# In[20]:


# # Skip this cell if the folders are already created
# # Creating a total of 8 classifications of age groups
class_folder_names = ['0-9','10-19','20-25','26-29','30-35','36-39','40-49','50+']

# Creating a new folder for all the classification data
os.mkdir('./classification_data/')

# Creating test and train folders in the classification_data folder
os.mkdir('./classification_data/Training')
os.mkdir('./classification_data/Validation')

# Using a for loop to create the age group folders in each train and test folder
for folder in ['Training', 'Validation']:
    for age_group in class_folder_names:
        os.mkdir(f'./classification_data/{folder}/{age_group}')

# # Printing directory list to see if process was finished correctly
# os.listdir('./classification_data/Training/')


# ## 2.
# 다운받은 데이터셋 폴더의 한글로된 폴더명 공백없이 바꿔주기

# In[ ]:


# import os

# def rename_folder(old_path, new_name):
#         # 기존 폴더의 전체 경로
#         old_path = os.path.abspath(old_path)

#         # 새로운 폴더명을 포함한 새로운 경로 생성
#         new_path = os.path.join(os.path.dirname(old_path), new_name)

#         # 폴더명 변경
#         os.rename(old_path, new_path)

# # 사용 예시
# old_folder_path1 = './118.안면 인식 에이징(aging) 이미지 데이터'
# new_folder_name1 = '118'
# rename_folder(old_folder_path1, new_folder_name1)

# old_folder_path2 = './118/01-1.정식개방데이터'
# new_folder_name2 = 'data'
# rename_folder(old_folder_path2, new_folder_name2)

# old_folder_path3 = './118/data/Training/01.원천데이터'
# new_folder_name3 = 'all_image'
# rename_folder(old_folder_path3, new_folder_name3)
# old_folder_path4 = './118/data/Training/02.라벨링데이터'
# new_folder_name4 = 'all_label'
# rename_folder(old_folder_path4, new_folder_name4)

# old_folder_path5 = './118/data/Validation/01.원천데이터'
# new_folder_name5 = 'all_image'
# rename_folder(old_folder_path5, new_folder_name5)
# old_folder_path6 = './118/data/Validation/02.라벨링데이터'
# new_folder_name6 = 'all_label'
# rename_folder(old_folder_path6, new_folder_name6)


# ## 3.
# '118' 폴더 아래에 있는 '정식개방데이터', '정식개방데이터' 아래에 있는 'Training' 폴더를 '118' 폴더 아래로 바로 옮겨주기

# In[ ]:


# import os
# import shutil

# def move_subfolders(source_folder, destination_folder):
#     # source_folder 안의 모든 하위 폴더 목록을 얻음
#     subfolders = [f.path for f in os.scandir(source_folder) if f.is_dir()]

#     # 각 하위 폴더를 destination_folder로 이동
#     for subfolder in subfolders:
#         # 새로운 목적지 경로 생성
#         new_destination = os.path.join(destination_folder, os.path.basename(subfolder))

#         # 폴더 이동
#         shutil.move(subfolder, new_destination)

# source_folder = "./118"
# destination_folder1 = "./118/data/Training"
# move_subfolders(source_folder, destination_folder1)

# destination_folder2 = "./118/data/Validation"
# move_subfolders(source_folder, destination_folder2)


# (디렉토리 구조) : capstone
# * face_data    
#     * 118 
#         * Training 
#             * all_image
#             * all_label
#         * Validation
#             * all_image
#             * all_label 
# * classification_data (-> T/V -> 8개 클래스)
# * text                
# + [.ipynb notbook]

# ## 4-1.
# Training과 Validation dataset에서 파일명 추출해서 모두 한 개의 text file에 저장

# In[4]:


# def save_png_filenames_to_txt(root_folder, output_txt_path):
#     with open(output_txt_path, 'w') as txt_file:
#         # root_folder 밑의 모든 폴더와 파일에 대해 재귀적으로 탐색
#         for foldername, subfolders, filenames in os.walk(root_folder):
#             for filename in filenames:
#                 # 파일이 .png로 끝나는 경우에만 파일명을 텍스트 파일에 추가
#                 if filename.lower().endswith('.png'):
#                     # full_path = os.path.join(foldername, filename)
#                     full_path = os.path.join(filename)
#                     txt_file.write(f"{full_path}\n")

# def general_path(train_or_test):
#     root = f"./118/{train_or_test}/all_image"
#     output = f"./118/text/{train_or_test}_filenames.txt"
#     return root, output

# # 파일 저장
# save_png_filenames_to_txt(*general_path("Training"))
# save_png_filenames_to_txt(*general_path("Validation"))

# # save_png_filenames_to_txt(*general_path("final_test"))


# ## 4-2.
# Training_filename과 Validation_filename에서 나이 추출해서 모두 한 개의 text file에 저장

# In[6]:


# def extract_and_save_words(input_file_path, output_file_path, target_word_index=2):
#     with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
#         # 파일의 각 줄을 읽어오기
#         for line in input_file:
#             # 줄을 공백 또는 다른 구분자로 나누기
#             words = line.strip().split('_')

#             # 특정 단어 추출 및 새로운 파일에 쓰기
#             if len(words) > target_word_index:
#                 target_word = words[target_word_index]
#                 output_file.write(f"{target_word}\n")

# def general_txt(train_or_test):
#     root = f"./text/{train_or_test}_sort.txt"
#     output = f"./text/{train_or_test}_age.txt"
#     return root, output

# # 파일 저장
# extract_and_save_words(*general_txt('Training'))
# extract_and_save_words(*general_txt('Validation'))

# extract_and_save_words(*general_txt('final_test'))


# ## 5.
# 파일 압축풀기

# In[ ]:


# import os
# from zipfile import ZipFile

# def unzip_all_zips_in_folders(parent_folder):
#     # 부모 폴더 아래의 모든 하위 폴더 가져오기
#     subfolders = [f.path for f in os.scandir(parent_folder) if f.is_dir()]

#     # 각 하위 폴더에 있는 .zip 파일 압축 해제
#     for folder in subfolders:
#         zip_files = [f for f in os.listdir(folder) if f.endswith('.zip')]
#         for zip_file in zip_files:
#             zip_file_path = os.path.join(folder, zip_file)
#             with ZipFile(zip_file_path, 'r') as zip_ref:
#                 zip_ref.extractall(folder)

# parent_folder_to_process = './118/'
# unzip_all_zips_in_folders(parent_folder_to_process)


# ## 6.
# [USE ONLY ONCE] 여러개의 폴더 안에 나뉘어 담겨있는 JSON파일들을 한 폴더 안으로 이동하기
# 
# [USE ONLY ONCE] 여러개의 폴더 안에 나뉘어 담겨있는 사진 파일들을 한 폴더 안으로 이동하기

# In[137]:


# import os
# import shutil

# def move_all_json_files(source_folder, destination_folder):
#     # 대상 폴더가 없으면 생성
#     if not os.path.exists(destination_folder):
#         os.makedirs(destination_folder)

#     # source_folder 내의 모든 하위 폴더 및 파일 순회
#     for root, dirs, files in os.walk(source_folder):
#         for file in files:
#             if file.endswith('.json'):  # JSON 파일인 경우
#                 source_path = os.path.join(root, file)
#                 destination_path = os.path.join(destination_folder, file)
#                 # 새로운 폴더로 JSON 파일 이동
#                 shutil.move(source_path, destination_path)

# def general_copy_files(train_or_test, data_type):
#     source_folder_path = f"./118/{train_or_test}/all_{data_type}"
#     destination_folder_path = f"./118/{train_or_test}/all_{data_type}_unzipped"
    
#     return source_folder_path, destination_folder_path

# # 여러개의 폴더 안에 나뉘어 담겨있는 JSON파일들을 한 폴더 안으로 복사하기
# move_all_json_files(*general_copy_files("Training", "label"))
# move_all_json_files(*general_copy_files("Validation", "label"))

# def move_png_files(source_folder, destination_folder):
#     # 대상 폴더가 없으면 생성
#     if not os.path.exists(destination_folder):
#         os.makedirs(destination_folder)

#     # 모든 하위 폴더에서 PNG 파일 찾기
#     for root, dirs, files in os.walk(source_folder):
#         for file in files:
#             if file.lower().endswith('.png'):
#                 source_path = os.path.join(root, file)
#                 destination_path = os.path.join(destination_folder, file)
#                 # 파일 이동
#                 shutil.move(source_path, destination_path)

# # 여러개의 폴더 안에 나뉘어 담겨있는 사진 파일들을 한 폴더 안으로 이동하기
# move_png_files(*general_copy_files("Training", "image"))
# move_png_files(*general_copy_files("Validation", "image"))


# ## 7.
# [USE ONLY ONCE] all_label 폴더안의 JSON파일들에 저장된 box 좌표대로 all_image 폴더안의 PNG파일들 자르기

# In[ ]:


# import os
# import json
# from PIL import Image

# def crop_and_save_images(json_folder_path, image_folder_path, output_folder_path):
#     # output_folder가 없으면 생성
#     if not os.path.exists(output_folder_path):
#         os.makedirs(output_folder_path)

#     # json 폴더 내의 모든 json 파일 순회
#     json_files = sorted([f for f in os.listdir(json_folder_path) if f.endswith('.json')])
#     for json_file in json_files:
#         json_path = os.path.join(json_folder_path, json_file)

#         # JSON 파일에서 이미지 영역 좌표 읽어오기
#         with open(json_path, 'r') as json_data:
#             data = json.load(json_data)
#             coordinates = data['annotation'][0]['box']

#         box = (coordinates['x'], coordinates['y'], 
#                coordinates['x'] + coordinates['w'], coordinates['y'] + coordinates['h'])

#         # 이미지 파일 경로 구성
#         image_file = os.path.splitext(json_file)[0] + '.png'
#         image_path = os.path.join(image_folder_path, image_file)

#         # PIL 이미지 열기
#         img = Image.open(image_path)

#         # 이미지를 주어진 좌표로 잘라내기
#         cropped_img = img.crop(box)

#         # 잘라낸 이미지를 새로운 폴더로 저장
#         output_path = os.path.join(output_folder_path, image_file)
#         cropped_img.save(output_path)

# def general_crop(train_or_test):
#     path_list = {
#         'json_folder_path': f"./118/{train_or_test}/all_label_unzipped",  # JSON 파일이 있는 폴더 경로
#         'image_folder_path': f"./118/{train_or_test}/all_image_unzipped",  # 이미지 파일이 있는 폴더 경로
#         'output_folder_path': f"./118/{train_or_test}/cropped_image"  # 저장할 이미지 폴더 경로
#     }
#     return path_list

# crop_and_save_images(**general_crop("Training"))
# crop_and_save_images(**general_crop("Validation"))


# ## 8.
# 이미지 resizing

# In[21]:


# import os
# import cv2

# def resize_images_in_folder(input_folder, output_folder, target_size=(224, 224)):
#     # 대상 폴더가 없으면 생성
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # 폴더 내의 모든 파일에 대해 반복
#     for filename in os.listdir(input_folder):
#         if filename.lower().endswith('.png'):
#             # 이미지 파일 경로
#             input_path = os.path.join(input_folder, filename)

#             # 이미지 로드
#             img = cv2.imread(input_path)

#             # 이미지 크기 조정
#             img_resized = cv2.resize(img, target_size)

#             # 출력 경로 및 파일명
#             output_path = os.path.join(output_folder, filename)

#             # 크기 조정된 이미지 저장
#             cv2.imwrite(output_path, img_resized)

# def general_resize(train_or_test):
#     input = f"./118/{train_or_test}/cropped_image"
#     output = f"./118/{train_or_test}/resized_image"
#     return input, output

# # 타겟 크기 설정
# target_size = (224, 224)

# # 이미지 크기 조정
# resize_images_in_folder(*general_resize('Training'), target_size)
# resize_images_in_folder(*general_resize('Validation'), target_size)


# ## 9.
# [USE ONLY ONCE] 
# 
# resized_image 폴더에 나이와 관계없이 모두 담겨있던 이미지들이 
# 
# classification_data 폴더안에 나이별로 구분해서 저장하기

# In[ ]:

# Loading in the txt file with the ages of each picture
train = pd.read_csv('./text/Training_age.txt', sep='\r', header=None)
test = pd.read_csv('./text/Validation_age.txt', sep='\r', header=None)
# final_test = pd.read_csv('./text/final_test_age.txt', sep='\r', header=None)

# Loading in the file_names of each picture
train['file_name'] = pd.read_csv('./text/Training_sort.txt', sep='\r', header=None)
test['file_name'] = pd.read_csv('./text/Validation_sort.txt', sep='\r', header=None)
# final_test['file_name'] = pd.read_csv('./final_test_filenames.txt', sep='\r', header=None)

# Renaming columns
train.columns = ['age','file_name']
test.columns = ['age','file_name']
# final_test.columns = ['age','file_name']


# In[25]:


# Function will classify each image appropriately, and then copy the original images from
# the MegaAge dataset into the appropriate folder for later modeling

def create_classification_folders(row, train_or_test):
    if row['age'] < 10:
        copy2(f"./118/{train_or_test}/resized_image/{row['file_name']}", f"./classification_data/{train_or_test}/0-9")
    elif row['age'] < 20:
        copy2(f"./118/{train_or_test}/resized_image/{row['file_name']}", f"./classification_data/{train_or_test}/10-19")
    elif row['age'] < 26:
        copy2(f"./118/{train_or_test}/resized_image/{row['file_name']}", f"./classification_data/{train_or_test}/20-25")
    elif row['age'] < 30:
        copy2(f"./118/{train_or_test}/resized_image/{row['file_name']}", f"./classification_data/{train_or_test}/26-29")
    elif row['age'] < 36:
        copy2(f"./118/{train_or_test}/resized_image/{row['file_name']}", f"./classification_data/{train_or_test}/30-35")
    elif row['age'] < 40:
        copy2(f"./118/{train_or_test}/resized_image/{row['file_name']}", f"./classification_data/{train_or_test}/36-39")
    elif row['age'] < 50:
        copy2(f"./118/{train_or_test}/resized_image/{row['file_name']}", f"./classification_data/{train_or_test}/40-49")
    else:
        copy2(f"./118/{train_or_test}/resized_image/{row['file_name']}", f"./classification_data/{train_or_test}/50+")

# Run the function on each row in Training
for index, row in train.iterrows():
    create_classification_folders(row, "Training")

# Run the function on each row in Validation
for index, row in test.iterrows():
    create_classification_folders(row, "Validation")

# Run the function on each row in final_test
# for index, row in final_test.iterrows():
#     create_classification_folders(row, "final_test")

