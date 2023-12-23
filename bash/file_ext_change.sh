#!/bin/bash

directory="/data/seanoh/capstone/face_data/118/Validation/all_image_unzipped"

# 디렉토리로 이동
cd "$directory"

# .PNG 파일을 .png로 변경
for file in *.PNG; do
    mv -- "$file" "${file%.PNG}.png"
done

