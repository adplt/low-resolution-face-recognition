from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Activation, Concatenate


def with_dimension_reduction(input_shape, depth, with_pooling=False, pool_size=(2, 2), name='inception'):
    tower_0 = Conv2D(depth, (1, 1), padding='same', data_format='channels_last', activation='relu',
                     name='tower0_' + name)(input_shape)

    tower_1 = Conv2D(depth, (1, 1), padding='same', data_format='channels_last', activation='relu',
                     name='tower1a_' + name)(input_shape)
    tower_1 = Conv2D(depth, (3, 3), padding='same', data_format='channels_last', activation='relu',
                     name='tower1b_' + name)(tower_1)

    tower_2 = Conv2D(depth, (1, 1), padding='same', data_format='channels_last', activation='relu',
                     name='tower2a_' + name)(input_shape)
    tower_2 = Conv2D(depth, (5, 5), padding='same', data_format='channels_last', activation='relu',
                     name='tower2b_' + name)(tower_2)

    tower_3 = MaxPooling2D(data_format='channels_last', padding='same', strides=(1, 1), pool_size=(2, 2),
                           name='tower_3a' + name)(input_shape)
    tower_3 = Conv2D(depth, (1, 1), padding='same', data_format='channels_last',
                     name='tower3b_' + name)(tower_3)

    inception_merge = Concatenate(axis=3, name='concenation_' + name if not with_pooling else None)([tower_0, tower_1, tower_2, tower_3])
    inception_activation = Activation('relu')(inception_merge)
    
    inception = MaxPooling2D(data_format='channels_last', padding='same', strides=(1, 1),
                             pool_size=pool_size, name='pooling_' + name)(inception_activation)\
        if with_pooling else inception_activation
    return inception
