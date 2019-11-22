# 20 Epochs

Train on 60000 samples, validate on 10000 samples
Epoch 1/20

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
60000/60000 [==============================] - 9s 145us/step - loss: 0.4233 - acc: 0.8993 - val_loss: 0.1091 - val_acc: 0.9807
Epoch 2/20

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022744503.
60000/60000 [==============================] - 6s 97us/step - loss: 0.1769 - acc: 0.9535 - val_loss: 0.0680 - val_acc: 0.9859
Epoch 3/20

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018315018.
60000/60000 [==============================] - 6s 96us/step - loss: 0.1374 - acc: 0.9643 - val_loss: 0.0562 - val_acc: 0.9856
Epoch 4/20

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015329586.
60000/60000 [==============================] - 6s 95us/step - loss: 0.1186 - acc: 0.9699 - val_loss: 0.0397 - val_acc: 0.9920
Epoch 5/20

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013181019.
60000/60000 [==============================] - 6s 99us/step - loss: 0.1029 - acc: 0.9740 - val_loss: 0.0423 - val_acc: 0.9897
Epoch 6/20

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011560694.
60000/60000 [==============================] - 6s 100us/step - loss: 0.0910 - acc: 0.9764 - val_loss: 0.0292 - val_acc: 0.9924
Epoch 7/20

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010295127.
60000/60000 [==============================] - 6s 100us/step - loss: 0.0861 - acc: 0.9771 - val_loss: 0.0259 - val_acc: 0.9943
Epoch 8/20

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009279307.
60000/60000 [==============================] - 6s 96us/step - loss: 0.0763 - acc: 0.9792 - val_loss: 0.0256 - val_acc: 0.9934
Epoch 9/20

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008445946.
60000/60000 [==============================] - 6s 96us/step - loss: 0.0709 - acc: 0.9806 - val_loss: 0.0286 - val_acc: 0.9932
Epoch 10/20

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007749935.
60000/60000 [==============================] - 6s 96us/step - loss: 0.0665 - acc: 0.9813 - val_loss: 0.0218 - val_acc: 0.9939
Epoch 11/20

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007159905.
60000/60000 [==============================] - 6s 95us/step - loss: 0.0634 - acc: 0.9819 - val_loss: 0.0216 - val_acc: 0.9943
Epoch 12/20

Epoch 00012: LearningRateScheduler setting learning rate to 0.000665336.
60000/60000 [==============================] - 6s 99us/step - loss: 0.0590 - acc: 0.9827 - val_loss: 0.0248 - val_acc: 0.9935
Epoch 13/20

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006213753.
60000/60000 [==============================] - 6s 97us/step - loss: 0.0583 - acc: 0.9821 - val_loss: 0.0218 - val_acc: 0.9941
Epoch 14/20

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005828638.
60000/60000 [==============================] - 6s 96us/step - loss: 0.0547 - acc: 0.9834 - val_loss: 0.0229 - val_acc: 0.9939
Epoch 15/20

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005488474.
60000/60000 [==============================] - 6s 96us/step - loss: 0.0527 - acc: 0.9826 - val_loss: 0.0195 - val_acc: 0.9948
Epoch 16/20

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005185825.
60000/60000 [==============================] - 6s 96us/step - loss: 0.0514 - acc: 0.9836 - val_loss: 0.0206 - val_acc: 0.9944
Epoch 17/20

Epoch 00017: LearningRateScheduler setting learning rate to 0.000491481.
60000/60000 [==============================] - 6s 101us/step - loss: 0.0485 - acc: 0.9841 - val_loss: 0.0215 - val_acc: 0.9946
Epoch 18/20

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004670715.
60000/60000 [==============================] - 6s 101us/step - loss: 0.0473 - acc: 0.9841 - val_loss: 0.0208 - val_acc: 0.9948
Epoch 19/20

Epoch 00019: LearningRateScheduler setting learning rate to 0.0004449718.
60000/60000 [==============================] - 6s 99us/step - loss: 0.0456 - acc: 0.9850 - val_loss: 0.0202 - val_acc: 0.9952
Epoch 20/20

Epoch 00020: LearningRateScheduler setting learning rate to 0.000424869.
60000/60000 [==============================] - 6s 102us/step - loss: 0.0465 - acc: 0.9843 - val_loss: 0.0191 - val_acc: 0.9952

# model.evaluate(on test data)

[0.01905159520963207, 0.9952]

# Summary

1. The obtained accuracy on validation set is 99.52(>99.4).
2. The bias had to be removed so used "bias=False" in each layer.
3. Reduced the parameters from 18.6k to 13.6k, by changing the the numbers of channel in 2nd layer to 16.
4. The difference between validation accuracy and training accuracy was significant so had to reduce the value of dropout from 0.1 to 0.05.
