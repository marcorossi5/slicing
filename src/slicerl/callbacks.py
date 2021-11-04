import tensorflow as tf

class ExtendedTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, x_train, y_train, *args, **kwargs):
        super(ExtendedTensorBoard, self).__init__(*args, **kwargs)
        self.x_train = x_train
        self.y_train = y_train

    def _log_gradients(self, epoch):
        step = tf.cast(epoch, dtype=tf.int64)
        writer = self._train_writer
        # writer = self._get_writer(self._train_run_name)

        with writer.as_default(), tf.GradientTape() as g:
            # here we use test data to calculate the gradients
            _x_batch = self.x_train
            _y_batch = self.y_train

            g.watch(tf.convert_to_tensor(_x_batch))
            _y_pred = self.model(_x_batch)  # forward-propagation
            loss = self.model.loss(y_true=_y_batch, y_pred=_y_pred)  # calculate loss
            gradients = g.gradient(loss, self.model.trainable_weights)  # back-propagation

            # In eager mode, grads does not have name, so we get names from model.trainable_weights
            for weights, grads in zip(self.model.trainable_weights, gradients):
                tf.summary.histogram(
                        weights.name.replace(':', '_')+'_grads', data=grads, step=step)

        writer.flush()

    def on_epoch_end(self, epoch, logs=None):  
        # def on_train_batch_end(self, batch, logs=None):  
        # This function overwrites the on_epoch_end in tf.keras.callbacks.TensorBoard
        # but we do need to run the original on_epoch_end, so here we use the super function. 
        super(ExtendedTensorBoard, self).on_epoch_end(epoch, logs=logs)
        # super(ExtendedTensorBoard, self).on_train_batch_end(batch, logs=logs)
        if self.histogram_freq and epoch % self.histogram_freq == 0:
            self._log_gradients(epoch)