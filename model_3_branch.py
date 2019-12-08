from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Concatenate, Input
import os
import inception


# # Just disables the warning, doesn't enable AVX/FMA (no GPU)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_model(width, height):
    input_img = Input(shape=(width, height, 3))

    conv1_convolution = Conv2D(64, (7, 7), strides=2, data_format='channels_last', activation='relu', padding='same', name='conv1_convolution')(input_img)
    conv1 = MaxPooling2D(data_format='channels_last', padding='same', strides=2, pool_size=(2, 2), name='conv1')(conv1_convolution)

    conv2_convolution = Conv2D(192, (3, 3), strides=2, data_format='channels_last', activation='relu', padding='same', name='conv2_convolution')(conv1)
    conv2 = MaxPooling2D(data_format='channels_last', padding='same', strides=1, pool_size=(2, 2), name='conv2')(conv2_convolution)

    inception3a_activation = inception.with_dimension_reduction(conv2, 64, False, name='inception3a_activation')
    inception3 = inception.with_dimension_reduction(inception3a_activation, 120, True, name='inception3')

    ######################################################### Trunk ############################################################

    inception4a_activation = inception.with_dimension_reduction(inception3, 128, False, name='inception4a_activation_trunk')
    inception4e_activation = inception.with_dimension_reduction(inception4a_activation, 132, False, name='inception4e_activation_trunk')
    inception4 = inception.with_dimension_reduction(inception4e_activation, 208, True, name='inception4_trunk')
    inception5a_activation = inception.with_dimension_reduction(inception4, 208, False, name='inception5a_activation_trunk')
    inception5b_1 = inception.with_dimension_reduction(inception5a_activation, 256, True, name='inception5b_1_trunk')

    ######################################################### Branch 1 ##########################################################

    inception4b_activation = inception.with_dimension_reduction(inception3, 128, False, name='inception4b_activation_branch_1')
    inception4e_activation = inception.with_dimension_reduction(inception4b_activation, 132, False, name='inception4e_activation_branch_1')
    inception4 = inception.with_dimension_reduction(inception4e_activation, 208, True, name='inception4_branch_1')
    inception5a_activation = inception.with_dimension_reduction(inception4, 208, False, name='inception5a_activation_branch_1')
    inception5b_2 = inception.with_dimension_reduction(inception5a_activation, 256, True, name='inception5b_2_branch_1')

    ######################################################### Branch 2 ##########################################################

    inception4c_activation = inception.with_dimension_reduction(inception3, 128, False, name='inception4c_activation_branch_2')
    inception4e_activation = inception.with_dimension_reduction(inception4c_activation, 132, False, name='inception4e_activation_branch_2')
    inception4 = inception.with_dimension_reduction(inception4e_activation, 208, True, name='inception4_branch_2')
    inception5a_activation = inception.with_dimension_reduction(inception4, 208, False, name='inception5a_activation_branch_2')
    inception5b_3 = inception.with_dimension_reduction(inception5a_activation, 256, True, name='inception5b_3_branch_2')

    ################################################ Branch 3 --- addition #######################################################

    inception4d_activation = inception.with_dimension_reduction(inception3, 128, False, name='inception4d_activation_branch_3')
    inception4e_activation = inception.with_dimension_reduction(inception4d_activation, 132, False, name='inception4e_activation_branch_3')
    inception4 = inception.with_dimension_reduction(inception4e_activation, 208, True, name='inception4_branch_3')
    inception5a_activation = inception.with_dimension_reduction(inception4, 208, False, name='inception5a_activation_branch_3')
    inception5b_4 = inception.with_dimension_reduction(inception5a_activation, 256, True, name='inception5b_4_branch_3')

    merged_branch = Concatenate(axis=1)([
        inception5b_3,
        inception5b_4,
    ])

    merged = Concatenate(axis=1)([
        inception5b_1,
        inception5b_2,
        merged_branch
    ])

    return input_img, merged
