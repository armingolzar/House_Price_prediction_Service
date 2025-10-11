from src.data_loader import Data_Prepration
from src.model import House_Price_Prediction_Network

SAVE_MODEL = "models\\house_price_prediction_model.h5"
SAVE_FIG = "assets\\training_validation_curve.png"
BATCH_SIZE = 32
EPOCHS = 100


def main():

    print("🚀 Loading and preparing data from Housing.csv file.")
    data_train, data_test, label_train, label_test, price_range, mid_price = (
        Data_Prepration(".\\data\\Housing.csv")
    )
    print(
        f"✅ Data loaded: {data_train.shape[0]} training, {data_test.shape[0]} test samples"
    )

    print(
        "🧠 Building neural network model with BatchNormalization and Dropout for generalization."
    )
    house_price_predictor = House_Price_Prediction_Network.network()

    print("Compile the network")
    compied_model = House_Price_Prediction_Network.compile_model(house_price_predictor)
    print("Model compied ✅")

    print(f"🏋️ Starting training for {EPOCHS} epochs...")
    house_price_prediction_model, history = House_Price_Prediction_Network.train(
        compied_model,
        data_train,
        label_train,
        BATCH_SIZE,
        EPOCHS,
        data_test,
        label_test,
    )
    final_val_mae = history.history["val_mae"][-1]
    final_val_mae_unscaled = final_val_mae * price_range
    print(
        f"🎯 The unscaled MAE is : {int(final_val_mae_unscaled)}."
        f"This means each prediction on avarage has {int(final_val_mae_unscaled)}$ error in pricing."
    )
    error_percentage_on_median = (final_val_mae_unscaled / mid_price) * 100
    print(
        f"🎯 The percentage of MAE error on the median of the price is : {int(error_percentage_on_median)}"
    )

    print("\n 🚀 Start saving model...")
    House_Price_Prediction_Network.saving_model(
        house_price_prediction_model, SAVE_MODEL
    )

    print("📊 Saving training curves")
    House_Price_Prediction_Network.show_result(history, SAVE_FIG)


if __name__ == "__main__":
    main()
