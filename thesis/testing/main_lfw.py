from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.models import load_model
import os
import numpy as np

# Just disables the warning, doesn't enable AVX/FMA (no GPU)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print('LFW')

path_dataset_lr = '../lfw'
img_width_lr, img_height_lr = 20, 20
batch_size = 8

datagen = ImageDataGenerator(rescale=1./255)

testing_generator = datagen.flow_from_directory(
    str(path_dataset_lr + '/test'),
    target_size=(img_width_lr, img_height_lr),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

test_steps_per_epoch = np.math.ceil(testing_generator.samples / testing_generator.batch_size)


model = load_model('tbe_cnn_lfw.h5')
model.summary()

print('predictions: ')

predictions = model.predict_generator(testing_generator, steps=test_steps_per_epoch)

predicted_classes = np.argmax(predictions, axis=1)
class_labels = list(testing_generator.class_indices.keys())

print('confusion matrix: ', confusion_matrix(testing_generator.classes, predicted_classes))
print('classification report: ', classification_report(testing_generator.classes, predicted_classes, target_names=class_labels))
