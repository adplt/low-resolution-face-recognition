import os
import shutil
import inspect

curr_directory = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
dataset_dir = os.path.join(curr_directory, 'out_dir_1')

URL_PATH = 'ytd'

training_dir = os.path.join(URL_PATH, 'train')
validate_dir = os.path.join(URL_PATH, 'validate')
testing_dir = os.path.join(URL_PATH, 'test')

list_label = os.listdir(dataset_dir)

for idx in range(len(list_label)):
    label_path_in = os.path.join(dataset_dir, list_label[idx])
    files = os.listdir(label_path_in)

    print('label:' + list_label[idx])
    print('\n')

    n_photo_train = int(float('70'.strip('%')) / 100 * len(files)) + 1
    n_photo_validation = int(float('20'.strip('%')) / 100 * len(files))
    n_photo_test = int(float('10'.strip('%')) / 100 * len(files))

    print('n_photo_train: ', n_photo_train)
    print('n_photo_validation: ', n_photo_validation)
    print('n_photo_test: ', n_photo_test)
    print('\n')

    # print file
    start = 0
    finish = n_photo_train
    photo_train = files[start:finish]

    print('start photo_train: ', start)
    print('finish photo_train: ', finish)
    print('\n')

    if n_photo_validation == 0:
        start = 0
        finish = n_photo_train
    else:
        start = n_photo_train
        finish = int(n_photo_train + n_photo_validation)
    photo_val = files[start:finish]

    print('start photo_val: ', start)
    print('finish photo_val: ', finish)
    print('\n')

    if n_photo_test == 0:
        start = 0
        finish = n_photo_train
    else:
        start = int(n_photo_train + n_photo_validation)
        finish = len(files)

    photo_test = files[start:finish]

    print('start photo_test: ', start)
    print('finish photo_test: ', finish)
    print('\n')

    print('n_photo_train: ' + str(n_photo_train))
    print('n_photo_validation: ' + str(n_photo_validation))
    print('n_photo_test: ' + str(n_photo_test))
    print('\n')

    print('photo_train: ', photo_train)
    print('photo_val: ', photo_val)
    print('photo_test: ', photo_test)

    if os.path.exists(os.path.join(training_dir, list_label[idx])):
        shutil.rmtree(os.path.join(training_dir, list_label[idx]))
    if os.path.exists(os.path.join(validate_dir, list_label[idx])):
        shutil.rmtree(os.path.join(validate_dir, list_label[idx]))
    if os.path.exists(os.path.join(testing_dir, list_label[idx])):
        shutil.rmtree(os.path.join(testing_dir, list_label[idx]))

    os.makedirs(os.path.join(training_dir, list_label[idx]))
    os.makedirs(os.path.join(validate_dir, list_label[idx]))
    os.makedirs(os.path.join(testing_dir, list_label[idx]))

    for image in photo_train:
        shutil.copyfile(os.path.join(dataset_dir, list_label[idx], image), os.path.join(training_dir, list_label[idx], image))

    for image in photo_val:
        shutil.copyfile(os.path.join(dataset_dir, list_label[idx], image), os.path.join(validate_dir, list_label[idx], image))

    for image in photo_test:
        shutil.copyfile(os.path.join(dataset_dir, list_label[idx], image), os.path.join(testing_dir, list_label[idx], image))

print('finish split dataset ', URL_PATH, '\n')
