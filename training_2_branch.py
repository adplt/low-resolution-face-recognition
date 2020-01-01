import model_2_branch
import common_function
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import SGD
import os
import asyncio
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.models import load_model, Model

# Just disables the warning, doesn't enable AVX/FMA (no GPU)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

epochs = 40
l_rate = 1.0e-4
decay = l_rate / epochs
sgd = SGD(lr=l_rate, momentum=0.9, decay=decay, nesterov=False)
batch_size = 32
img_width, img_height = 24, 24
path_data_set = './ytd'
input_img, merged = model_2_branch.get_model(img_width, img_height)
num_train_images = 424961  # training images: 424961  # total images: 605855
file_path = 'tbe_cnn_ytd_2_branch.h5'


datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)


############################################### Training Dataset #############################################################


async def training():
    if not os.path.exists(file_path):
        flatten = Flatten()(merged)
        dense = Dense(64)(flatten)
        activation = Activation('softmax')(dense)
        dropout = Dropout(0.5)(activation)
        dense = Dense(1591)(dropout)
        activation = Activation('softmax')(dense)
    
        base_model = Model(input_img, activation)
    else:
        base_model = load_model(file_path)
        # base_model.load_weights(file_path)
    
    base_model.summary()
    
    train_generator_lr = datagen.flow_from_directory(
        str(path_data_set + '/train'),
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator_lr = datagen.flow_from_directory(
        str(path_data_set + '/validate'),
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
    )
    
    print('training: ')

    base_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    checkpoint = ModelCheckpoint(
        file_path,
        monitor='val_loss',
        save_best_only=True,
        mode='auto',
        verbose=1
    )
    callbacks_list = [checkpoint]
    
    history = base_model.fit_generator(
        generator=train_generator_lr,
        steps_per_epoch=num_train_images // batch_size,
        epochs=epochs,
        validation_data=validation_generator_lr,
        validation_steps=800 // batch_size,
        callbacks=callbacks_list
    )
    
    common_func = common_function.CommonFunction()
    common_func.plot_training(history, 'TBE-CNN (SGD)')


if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.run_until_complete(training())
