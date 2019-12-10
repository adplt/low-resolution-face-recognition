from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import load_model
import os
import numpy as np
import tensorflow as tf
import keras.backend as K


# Just disables the warning, doesn't enable AVX/FMA (no GPU)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print('YTD')

path_dataset_lr = './ytd'
img_width_lr, img_height_lr = 24, 24
batch_size = 32
num_train_images = 60363

datagen = ImageDataGenerator()  # rescale=1./255

testing_generator = datagen.flow_from_directory(
    str(path_dataset_lr + '/test'),
    target_size=(img_width_lr, img_height_lr),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_steps_per_epoch = np.math.ceil(testing_generator.samples / testing_generator.batch_size)


model = load_model('tbe_cnn_ytd.h5')
model.summary()

print('predictions: ')
# steps=test_steps_per_epoch
predictions = model.predict_generator(testing_generator, num_train_images // batch_size + 1)

predicted_classes = np.argmax(predictions, axis=1)
true_class_label = K.argmax(y_true, axis=-1)

cm = tf.confusion_matrix(testing_generator.classes, predicted_classes)

print('testing_generator: ', testing_generator.classes)
print('predicted_classes: ', predicted_classes)

diag = tf.linalg.tensor_diag_part(cm)

# Calculate the total number of data examples for each class
total_per_class = tf.reduce_sum(cm, axis=1)

acc_per_class = diag / tf.maximum(1, total_per_class)
nan_mask = tf.debugging.is_nan(acc_per_class)
x = tf.boolean_mask(acc_per_class, tf.logical_not(nan_mask))
uar = K.mean(x)

print('uar: ', uar)
