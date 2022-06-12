from tensorflow.keras import layers


class UnetDownsamplingConv(layers.Layer):
    def __init__(self, filter_count, max_pool):
        super().__init__()
        self.conv = layers.Conv3D(
            filter_count,
            kernel_size=(3, 3, 3),
            strides=(1, 1, 1),
            padding="same",
        )
        self.batch_norm = layers.BatchNormalization()
        self.activation = layers.Activation("relu")
        self.max_pool = layers.MaxPooling3D(pool_size=(2, 2, 2)) if max_pool else None

    def call(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        if self.max_pool:
            x = self.max_pool(x)
        return x


class UnetUpsamplingConv(layers.Layer):
    def __init__(self, filter_count):
        super().__init__()
        self.upsample = layers.UpSampling3D(size=(2, 2, 2))
        self.unet_conv = UnetDownsamplingConv(filter_count, False)

    def call(self, x1, x2):
        x2 = self.upsample(x2)
        x1_dims = x1.get_shape()
        x2_dims = x2.get_shape()
        channels_indexes = [1, 2, 3]
        dim_diffs = [(x2_dims[i] - x1_dims[i]) for i in channels_indexes]
        padding = [(diff // 2, diff // 2 + (diff % 2)) for diff in dim_diffs]
        x1 = layers.ZeroPadding3D(padding)(x1)
        x = layers.Concatenate(axis=4)([x1, x2])
        return self.unet_conv(x)


class UnetDeepSupervision(layers.Layer):
    def __init__(self, filter_count, scale_factor):
        super().__init__()
        self.conv = layers.Conv3D(
            filter_count, kernel_size=1, strides=1, padding="valid"
        )
        self.upsample = layers.UpSampling3D(size=scale_factor)

    def call(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x


class UnetGridGatingSignal(layers.Layer):
    def __init__(self, filter_count, kernel_size):
        super().__init__()
        self.conv = layers.Conv3D(
            filter_count, kernel_size, strides=(1, 1, 1), padding="valid"
        )
        self.batch_norm = layers.BatchNormalization()
        self.ac = layers.Activation("relu")

    def call(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.ac(x)
        return x
