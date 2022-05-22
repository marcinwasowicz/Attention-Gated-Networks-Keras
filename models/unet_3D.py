from tensorflow.keras import layers, models, optimizers

from .utils import dice_loss


class Unet3DFactory:
    def __init__(self, input_shape, num_classes, learning_rate, **kwargs):
        self.input_shape = tuple(input_shape) + (1,)
        self.num_classes = num_classes
        self.learning_rate = learning_rate

    def produce_unet(self):
        skeleton = [layers.Input(self.input_shape)]
        filter_counts = [16, 32, 64, 128, 256]

        # Downsampling
        for filter_count in filter_counts:
            for layer in self._conv_block(filter_count, True):
                layer = layer(skeleton[-1])
                skeleton.append(layer)

        # Remove last max pooling
        skeleton.pop(-1)

        # Upsampling
        convolution_ids = [1, 5, 9, 13]
        for upsample_id, filter_count in enumerate(reversed(filter_counts[:-1])):
            upsampling_layer = layers.UpSampling3D(size=(2, 2, 2))(skeleton[-1])
            skeleton.append(upsampling_layer)
            convolution_id = convolution_ids[-(upsample_id + 1)]
            skeleton.append(
                layers.Concatenate(axis=4)([skeleton[-1], skeleton[convolution_id]])
            )
            for layer in self._conv_block(filter_count, False):
                layer = layer(skeleton[-1])
                skeleton.append(layer)

        # Final convolution
        final_conv = layers.Conv3D(self.num_classes, 1, activation="softmax")(
            skeleton[-1]
        )
        skeleton.append(final_conv)

        model = models.Model(inputs=skeleton[0], outputs=skeleton[-1])
        model.compile(optimizer=optimizers.Adam(self.learning_rate), loss=dice_loss)
        return model

    def _conv_block(self, filter_count, max_pool):
        conv_block = [
            layers.Conv3D(
                filter_count,
                kernel_size=(3, 3, 3),
                strides=(1, 1, 1),
                padding="same",
            ),
            layers.BatchNormalization(),
            layers.Activation("relu"),
        ]

        if max_pool:
            conv_block.append(layers.MaxPool3D(pool_size=(2, 2, 2)))
        return conv_block
