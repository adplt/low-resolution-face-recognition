# from keras.layers import Conv2D, MaxPooling2D, Activation, Concatenate
#
#
# def with_dimension_reduction(input_shape, depth, with_pooling=False):
#     tower_0 = Conv2D(depth, (1, 1), padding='same', data_format='channels_last', activation='relu')(input_shape)
#
#     tower_1 = Conv2D(depth, (1, 1), padding='same', data_format='channels_last', activation='relu')(input_shape)
#     tower_1 = Conv2D(depth, (3, 3), padding='same', data_format='channels_last', activation='relu')(tower_1)
#
#     tower_2 = Conv2D(depth, (1, 1), padding='same', data_format='channels_last', activation='relu')(input_shape)
#     tower_2 = Conv2D(depth, (5, 5), padding='same', data_format='channels_last', activation='relu')(tower_2)
#
#     tower_3 = MaxPooling2D(data_format='channels_last', padding='same', strides=(1, 1), pool_size=(2, 2))(input_shape)
#     tower_3 = Conv2D(depth, (1, 1), padding='same', data_format='channels_last')(tower_3)
#
#     inception_merge = Concatenate(axis=3)([tower_0, tower_1, tower_2, tower_3])
#     inception_activation = Activation('relu')(inception_merge)
#     inception = MaxPooling2D(data_format='channels_last', pool_size=(2, 2))(inception_activation)\
#         if with_pooling else inception_activation
#     return inception
