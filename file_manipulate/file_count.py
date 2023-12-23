import os

def count_files_in_directory(directory):
    total_files = 0

    for root, dirs, files in os.walk(directory):
        total_files += len(files)

    return total_files

path= "/data/seanoh/capstone/face_data/classification_data/Validation/"
count= count_files_in_directory(path)
print(f"{count}")

