import os
import inspect

curr_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
file_directory = os.path.join(curr_directory, './ytd/validate')

list_label = os.listdir(file_directory)

i = 0

while i < len(list_label):
    print('list_label ke i: ', list_label[i])
    if list_label[i] is '.DS_Store':
        os.remove(file_directory + '/.DS_Store')
    path = os.path.join(file_directory, list_label[i])
    print('len path: ', len(os.listdir(path)))
    if len(os.listdir(path)) == 0:
        print('remove: ', path)
        os.rmdir(path)
    i += 1

print('finish remove empty folder')
