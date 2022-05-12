from tensorflow.python.keras import layers, backend

def attention_gate_2d(x, g, inter_channel):
    shape_x = backend.int_shape(x)
    shape_g = backend.int_shape(g)

    theta_x = layers.Conv2D(inter_channel, (2, 2), strides=(2, 2), padding='same')(x)
    shape_theta_x = backend.int_shape(theta_x)

    phi_g = layers.Conv2D(inter_channel, (1, 1), padding='same')(g)
    upsample_g = layers.Conv2DTranspose(inter_channel, (3, 3), strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]), padding='same')(phi_g)

    f = layers.Activation('relu')(layers.add([theta_x, upsample_g]))

    psi_f = layers.Conv2D(1, (1, 1), padding='same')(f)

    rate = layers.Activation('sigmoid')(psi_f)
    shape_rate = backend.int_shape(rate)

    upsample_rate = layers.UpSampling2D(size=(shape_x[1] // shape_rate[1], shape_x[2] // shape_rate[2]))(rate)
    upsample_rate = layers.Lambda(lambda x, repnum: backend.repeat_elements(x, repnum, axis=3), arguments={'repnum': shape_x[3]})(upsample_rate)

    att_x = layers.multiply([x, upsample_rate])

    result = layers.Conv2D(shape_x[3], (1, 1), padding='same')(att_x)

    return layers.BatchNormalization()(result)
