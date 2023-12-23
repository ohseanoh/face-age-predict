#!/bin/bash

# 파일이 있는 디렉토리로 변경
cd /data/seanoh/capstone/face_data/118/Validation/all_label_unzipped/

# 해당 패턴과 일치하는 파일 찾기
for file in *_d.json *_f.json; do
    # 새로운 파일 이름 생성
    if [[ $file == *_d.json ]]; then
        new_name="${file%_d.json}_D.json"
    elif [[ $file == *_f.json ]]; then
        new_name="${file%_f.json}_F.json"
    fi

    # 파일 이동 및 이름 변경
    mv "$file" "$new_name"
done

