from keras.callbacks import Callback
import mlflow


class MetricsHistory(Callback):
    def on_train_begin(self, logs=None):
        self.training_loss = dict()
        self.validation_loss= dict()

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
        print("End epoch {} of training; got log keys: {}".format(epoch, keys))

        print(
            "The average loss for epoch {} is {:7.2f} ".format(
                epoch, logs["recon_loss"]
            )
        )

        try:
            self.training_loss[f"training loss"] = logs["loss"]
        except KeyError:
            self.training_loss[f"training loss"] = logs["recon_loss"]

        try:
            self.validation_loss[f"validation loss"] = logs["val_loss"]
        except KeyError:
            self.validation_loss[f"validation loss"] = logs["val_recon_loss"]

        mlflow.log_metrics({**self.training_loss, **self.validation_loss}, step=epoch)
