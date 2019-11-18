from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Activation, Concatenate


def with_dimension_reduction(input_shape, depth, with_pooling=False, pool_size=(2, 2), name='inception'):
    tower_0 = Conv2D(depth, (1, 1), padding='same', data_format='channels_last', activation='softmax')(input_shape)

    tower_1 = Conv2D(depth, (1, 1), padding='same', data_format='channels_last', activation='softmax')(input_shape)
    tower_1 = Conv2D(depth, (3, 3), padding='same', data_format='channels_last', activation='softmax')(tower_1)

    tower_2 = Conv2D(depth, (1, 1), padding='same', data_format='channels_last', activation='softmax')(input_shape)
    tower_2 = Conv2D(depth, (5, 5), padding='same', data_format='channels_last', activation='softmax')(tower_2)

    tower_3 = MaxPooling2D(data_format='channels_last', padding='same', strides=(1, 1), pool_size=(2, 2))(input_shape)
    tower_3 = Conv2D(depth, (1, 1), padding='same', data_format='channels_last')(tower_3)

    inception_merge = Concatenate(axis=3, name=name if not with_pooling else None)([tower_0, tower_1, tower_2, tower_3])
    inception_activation = Activation('softmax')(inception_merge)
    
    inception = MaxPooling2D(data_format='channels_last', padding='same', strides=(1, 1),
                             pool_size=pool_size, name=name)(inception_activation)\
        if with_pooling else inception_activation
    return inception
