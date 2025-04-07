import os

# 替换为您的目标文件夹路径
directory_path = '/data/zhchen/Mayo2016_2d/test/full_1mm'

# 替换为您想要创建的文本文件的路径
output_file_path = '/data/zhchen/Mayo2016_2d/mayo2016_test.txt'

# 遍历文件夹中的所有文件，并将文件名写入文本文件
with open(output_file_path, 'w') as file:
    for filename in os.listdir(directory_path):
        if os.path.isfile(os.path.join(directory_path, filename)):
            file.write(filename + '\n')