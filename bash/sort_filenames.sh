
# 파일 경로
input_file="/data/seanoh/capstone/face_data/text/Validation_filenames.txt"
output_file="/data/seanoh/capstone/face_data/text/Validation_sort.txt"

# 정렬 및 저장
sort -k1.1,1.4 -n "$input_file" > "$output_file"

echo "파일이 정렬되어 $output_file 에 저장되었습니다."

