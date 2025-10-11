from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os


class House_Price_Prediction_Network:

    def __init__(self):
        pass

    @staticmethod
    def network():

        input_layer = Input(shape=(12,), name="input_layer")

        dense1 = Dense(8, activation="relu", name="dense1")(input_layer)

        drop1 = Dropout(0.1, name="drop1")(dense1)

        dense2 = Dense(4, activation="relu", name="dense2")(drop1)

        output = Dense(1, activation="linear", name="output")(dense2)

        model = Model(inputs=input_layer, outputs=output)

        model.summary()

        return model

    @staticmethod
    def compile_model(model, lr=0.01):

        opt = Adam(lr)
        model.compile(optimizer=opt, loss="mse", metrics=["mae"])

        return model

    @staticmethod
    def train(model, data, label, batch_size, epoch, data_val, label_val):

        history = model.fit(
            data,
            label,
            batch_size,
            epoch,
            validation_data=(data_val, label_val),
            shuffle=True,
        )

        return model, history

    @staticmethod
    def saving_model(model, path):

        model.save(os.path.join(os.getcwd(), path))
        print("Model saved in : models foldar")

    @staticmethod
    def show_result(history, path):

        plt.figure(figsize=(8, 4))
        plt.subplot(1, 1, 1)
        plt.plot(history.history["mae"], label="Training Curve")
        plt.plot(history.history["val_mae"], label="Validation Curve")
        plt.title("Model Training Result")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(os.getcwd(), path))
        plt.close()

        print("Figure saved in : assets folder")
