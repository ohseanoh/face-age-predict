#!/bin/bash

# 원본 폴더와 대상 폴더를 지정
target="/data/seanoh/capstone/face_data/classification_data/Validation/36-39/"
destination="/data/seanoh/capstone/face_data/classification_data/Validation/30-35/"

# 원본 폴더로 이동
cd "$target" || exit

# 해당 폴더 아래의 모든 파일을 대상 폴더로 이동
find . -type f -exec mv -i {} "$destination" \;
