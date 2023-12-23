#!/bin/bash

# 디렉토리 경로 설정
target_directory="/data/seanoh/capstone/face_data/118/Validation/all_label/"

# 디렉토리로 이동
cd "$target_directory"

# .zip 파일을 찾아 압축 해제
for zip_file in *.zip; do
    unzip "$zip_file" -d "${zip_file%.zip}"
    # 압축 해제가 완료된 파일은 삭제할 경우
    rm "$zip_file"
done

