import matplotlib.pyplot as plt


class CommonFunction:
    def __init__(self):
        print('Common Function')
    
    def plot_training(self, history, title='Training and Validation Accuracy'):
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(acc))
        
        plt.plot(epochs, acc, 'g.', label='accuracy')
        plt.plot(epochs, val_acc, 'g', label='val_acc')
        plt.title(title)
        
        # plt.figure()
        
        plt.plot(epochs, loss, 'r.', label='loss')
        plt.plot(epochs, val_loss, 'r-', label='val_loss')
        plt.legend()
        plt.show()
        
        plt.savefig('acc_vs_epochs.png')


# from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Activation, Concatenate


def with_dimension_reduction(input_shape, depth, with_pooling=False, pool_size=(2, 2), name='inception'):
    tower_0 = Conv2D(depth, (1, 1), padding='same', data_format='channels_last', activation='relu')(input_shape)
    
    tower_1 = Conv2D(depth, (1, 1), padding='same', data_format='channels_last', activation='relu')(input_shape)
    tower_1 = Conv2D(depth, (3, 3), padding='same', data_format='channels_last', activation='relu')(tower_1)
    
    tower_2 = Conv2D(depth, (1, 1), padding='same', data_format='channels_last', activation='relu')(input_shape)
    tower_2 = Conv2D(depth, (5, 5), padding='same', data_format='channels_last', activation='relu')(tower_2)
    
    tower_3 = MaxPooling2D(data_format='channels_last', padding='same', strides=(1, 1), pool_size=(2, 2))(input_shape)
    tower_3 = Conv2D(depth, (1, 1), padding='same', data_format='channels_last')(tower_3)
    
    inception_merge = Concatenate(axis=3, name=name if not with_pooling else None)([tower_0, tower_1, tower_2, tower_3])
    inception_activation = Activation('relu')(inception_merge)
    
    inception = MaxPooling2D(data_format='channels_last', padding='same', strides=(1, 1), pool_size=pool_size,
                             name=name)(inception_activation) \
        if with_pooling else inception_activation
    return inception


from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Concatenate  # , Input, Lambda
import os
from tensorflow.python.keras import backend as K
# from tensorflow.python.keras.layers import Dense, Flatten, Dropout
# from tensorflow.python.keras.models import Model

# from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input

# # Just disables the warning, doesn't enable AVX/FMA (no GPU)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
img_width_lr, img_height_lr = 24, 24
img_width_hr, img_height_hr = 192, 192
FC_LAYERS = [1024, 1024]
dropout = 0.5


def build_fine_tune_model(base_model, dropout, fc_layers, num_classes):
    for layer in base_model.layers:
        layer.trainable = False
    
    x = base_model.output
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)
    
    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x)
    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    
    return finetune_model


# using euclidean distance
def couple_mapping(vector):
    x, y = vector
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))


def get_model():
    input_img_lr = Input(shape=(img_height_lr, img_width_lr, 3))
    input_img_hr = Input(shape=(img_height_hr, img_width_hr, 3))
    
    # base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(img_height_lr, img_width_lr, 3))
    # fine_tune_model = build_fine_tune_model(
    #     base_model,
    #     dropout=dropout,
    #     fc_layers=FC_LAYERS,
    #     num_classes=5724
    # )
    
    conv1_convolution = Conv2D(64, (7, 7), strides=2, data_format='channels_last', activation='relu', padding='same',
                               name='conv1_convolution')(input_img_hr)
    # print('conv1_convolution: ' + str(K.int_shape(conv1_convolution)))
    conv1 = MaxPooling2D(data_format='channels_last', padding='same', strides=2, pool_size=(2, 2), name='conv1')(
        conv1_convolution)
    # print('conv1: ' + str(K.int_shape(conv1)))
    
    conv2_convolution = Conv2D(192, (3, 3), strides=2, data_format='channels_last', activation='relu', padding='same',
                               name='conv2_convolution')(conv1)
    # print('conv2_convolution: ' + str(K.int_shape(conv2_convolution)))
    conv2 = MaxPooling2D(data_format='channels_last', padding='same', strides=1, pool_size=(2, 2), name='conv2')(
        conv2_convolution)
    # print('conv2: ' + str(K.int_shape(conv2)))
    
    inception3a_activation = with_dimension_reduction(input_img_lr, 64, False, name='inception3a_activation')
    inception3 = with_dimension_reduction(inception3a_activation, 120, True, name='inception3')
    
    ######################################################### Trunk ############################################################
    
    inception4a_activation = with_dimension_reduction(conv2, 128, False, name='inception4a_activation_trunk')
    inception4e_activation = with_dimension_reduction(inception4a_activation, 132, False,
                                                      name='inception4e_activation_trunk')
    inception4 = with_dimension_reduction(inception4e_activation, 208, True, name='inception4_trunk')
    inception5a_activation = with_dimension_reduction(inception4, 208, False, name='inception5a_activation_trunk')
    inception5b_1 = with_dimension_reduction(inception5a_activation, 256, True, name='inception5b_1_trunk')
    
    ######################################################### Branch 1 ##########################################################
    
    inception4b_activation = with_dimension_reduction(inception3, 128, False, name='inception4b_activation_branch_1')
    inception4e_activation = with_dimension_reduction(inception4b_activation, 132, False,
                                                      name='inception4e_activation_branch_1')
    inception4 = with_dimension_reduction(inception4e_activation, 208, True, name='inception4_branch_1')
    inception5a_activation = with_dimension_reduction(inception4, 208, False, name='inception5a_activation_branch_1')
    inception5b_2 = with_dimension_reduction(inception5a_activation, 256, True, name='inception5b_2_branch_1')
    
    ######################################################### Branch 2 ##########################################################
    
    inception4c_activation = with_dimension_reduction(inception3, 128, False, name='inception4c_activation_branch_2')
    inception4e_activation = with_dimension_reduction(inception4c_activation, 132, False,
                                                      name='inception4e_activation_branch_2')
    inception4 = with_dimension_reduction(inception4e_activation, 208, True, name='inception4_branch_2')
    inception5a_activation = with_dimension_reduction(inception4, 208, False, name='inception5a_activation_branch_2')
    inception5b_3 = with_dimension_reduction(inception5a_activation, 256, True, name='inception5b_3_branch_2')
    
    ################################################ Branch 3 --- addition #######################################################
    
    inception4d_activation = with_dimension_reduction(inception3, 128, False, name='inception4d_activation_branch_3')
    inception4e_activation = with_dimension_reduction(inception4d_activation, 132, False,
                                                      name='inception4e_activation_branch_3')
    inception4 = with_dimension_reduction(inception4e_activation, 208, True, name='inception4_branch_3')
    inception5a_activation = with_dimension_reduction(inception4, 208, False, name='inception5a_activation_branch_3')
    inception5b_4 = with_dimension_reduction(inception5a_activation, 256, True, name='inception5b_4_branch_3')
    
    # couple_mapping_1 = Lambda(couple_mapping, name='couple_mapping_1')([inception5b_3, inception5b_4])
    # couple_mapping_2 = Lambda(couple_mapping, name='couple_mapping_2')([inception5b_1, inception5b_4])
    # couple_mapping_3 = Lambda(couple_mapping, name='couple_mapping_3')([inception5b_1, inception5b_2])
    
    # print('couple_mapping_1: ' + str(K.int_shape(couple_mapping_1)))
    # print('couple_mapping_2: ' + str(K.int_shape(couple_mapping_2)))
    # print('couple_mapping_3: ' + str(K.int_shape(couple_mapping_3)))
    
    merged_branch = Concatenate(axis=1)([
        inception5b_3,
        inception5b_4,
        # couple_mapping_last
    ])
    
    merged = Concatenate(axis=1)([
        # couple_mapping_1,
        # couple_mapping_2,
        # couple_mapping_3
        inception5b_1,
        inception5b_2,
        # couple_mapping_last,
        merged_branch
    ])
    
    return input_img_lr, input_img_hr, merged


from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import SGD
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation, Input
from tensorflow.python.keras.models import Model
import os
import asyncio
import nest_asyncio
import zipfile as zf

# Just disables the warning, doesn't enable AVX/FMA (no GPU)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

epochs = 10
l_rate = 0.01
decay = l_rate / epochs
sgd = SGD(lr=l_rate, momentum=0.9, decay=decay, nesterov=False)
batch_size = 8
img_width_lr, img_height_lr = 24, 24
img_width_hr, img_height_hr = 192, 192
path_dataset_lr = 'lfw'
input_img_lr, input_img_hr, merged = get_model()
num_train_images = 136117

datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)


def generate_multiple_input():
    train_generator_lr = datagen.flow_from_directory(
        str(path_dataset_lr + '/train'),
        target_size=(img_width_lr, img_height_lr),
        batch_size=batch_size,
        class_mode='categorical')
    
    train_generator_hr = datagen.flow_from_directory(
        str(path_dataset_lr + '/train'),
        target_size=(img_width_hr, img_height_hr),
        batch_size=batch_size,
        class_mode='categorical')
    
    while True:
        X1i = train_generator_lr.next()
        X2i = train_generator_hr.next()
        yield [X1i[0], X2i[0]], X2i[1]  # Yield both images and their mutual label


############################################### Training Dataset #############################################################


async def training():
    flatten = Flatten()(merged)
    dense = Dense(64)(flatten)
    activation = Activation('relu')(dense)
    dropout = Dropout(0.5)(activation)
    dense = Dense(5724)(dropout)
    activation = Activation('sigmoid')(dense)
    
    model = Model([input_img_lr, input_img_hr], activation)
    model.summary()
    
    validation_generator_lr = datagen.flow_from_directory(
        str(path_dataset_lr + '/validate'),
        target_size=(img_width_lr, img_height_lr),
        batch_size=batch_size,
        class_mode='categorical')
    
    print('training: ')
    
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    history = model.fit_generator(
        generator=generate_multiple_input(),  # [train_generator_lr, train_generator_lr],
        steps_per_epoch=num_train_images // batch_size,
        epochs=epochs,
        validation_data=validation_generator_lr,
        validation_steps=800 // batch_size)
    
    model_json = model.to_json()
    with open('tbe_cnn.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('tbe_cnn.h5')
    print('Saved model to disk')
    
    common_func = CommonFunction()
    common_func.plot_training(history, 'TBE-CNN Training and Validation')


if __name__ == '__main__':
    if not os.path.exists('lfw'):
        os.makedirs('lfw')
    files = zf.ZipFile("lfw.zip", 'r')
    files.extractall('lfw')
    files.close()
    nest_asyncio.apply()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(training())
