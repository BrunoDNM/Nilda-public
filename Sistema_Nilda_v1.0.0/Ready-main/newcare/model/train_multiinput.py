import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

from keras import regularizers
from keras.layers import (
    Input, Conv2D, BatchNormalization,
    Dense, Dropout, Concatenate, MaxPooling2D, GlobalAveragePooling2D,
)
from keras.models import Model

NEWCARE_DIR = str(Path(__file__).resolve().parents[1])
if NEWCARE_DIR not in sys.path:
    sys.path.insert(0, NEWCARE_DIR)

from model.training import run_training


def build_multi_input_model(img_shape=(128, 128, 1), num_features=4):
    input_img = Input(shape=img_shape, name="image_input")
    x = input_img

    x = Conv2D(16, (3, 3), activation="relu", padding="same",
               kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.1)(x)

    x = Conv2D(32, (3, 3), activation="relu", padding="same",
               kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    x = Dropout(0.1)(x)

    x = Conv2D(64, (3, 3), activation="relu", padding="same",
               kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)

    # GAP em vez de Flatten pra caber no servidor com pouca memoria
    x = GlobalAveragePooling2D()(x)
    x = Dense(32, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = Dropout(0.5)(x)
    x = Dense(16, activation="relu", name="image_branch")(x)

    input_feat = Input(shape=(num_features,), name="feature_input")
    y = Dense(8, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(input_feat)
    y = Dropout(0.3)(y)
    y = Dense(4, activation="relu", name="feature_branch")(y)

    combined = Concatenate(name="concat")([x, y])
    z = Dense(8, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(combined)
    z = Dropout(0.3)(z)
    output = Dense(1, activation="sigmoid", name="output")(z)

    model = Model(inputs=[input_img, input_feat], outputs=output)
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    run_training(
        build_multi_input_model, "NewCare", "multi", (128, 128),
        Path(__file__).resolve().parent / "dados",
    )


if __name__ == "__main__":
    main()
