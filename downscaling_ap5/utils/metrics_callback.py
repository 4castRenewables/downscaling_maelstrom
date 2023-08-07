from keras.callbacks import Callback
import mlflow


class MetricsHistory(Callback):
    def on_train_begin(self, logs={}):
        self.training_loss = dict()
        self.validation_loss= dict()

    def on_epoch_end(self, epoch, logs={}):
        try:
            self.training_loss[f"training loss epoch {epoch}"] = logs.get('loss')
        except KeyError:
            self.training_loss[f"training loss epoch {epoch}"] = logs.get('recon_loss')

        try:
            self.validation_loss[f"validation loss epoch {epoch}"] = logs.get('val_loss')
        except KeyError:
            self.validation_loss[f"validation loss epoch {epoch}"] = logs.get('val_recon_loss')

        mlflow.log_metrics({**self.training_loss, **self.validation_loss})