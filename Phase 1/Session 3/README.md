## Final validation Accuracy obtained

### 83.79

## Model Definition with size of Output Channel and Receptive Field

_Defining the model_

model = Sequential()
model.add(SeparableConv2D(48, kernel_size=(3,3), input_shape=(32, 32, 3))) #30X30X48 #RF=3X3
model.add(BatchNormalization())
#model.add(Convolution2D(48, 3, 3, , input_shape=(32, 32, 3)))
model.add(Activation('relu'))

model.add(SeparableConv2D(48, kernel_size=(3,3))) #28X28X48 #RF=5X5
model.add(BatchNormalization())
#model.add(Convolution2D(48, 3, 3))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2))) #14X14X48 #RF=6X6
model.add(Dropout(0.25))

model.add(SeparableConv2D(96, kernel_size=(3,3), border_mode='same')) #14X14X96 #RF=10X10
model.add(BatchNormalization())
#model.add(Convolution2D(96, 3, 3, border_mode='same'))
model.add(Activation('relu'))

model.add(SeparableConv2D(96, kernel_size=(3,3))) #12X12X96 #RF=14X14
model.add(BatchNormalization())
#model.add(Convolution2D(96, 3, 3))
model.add(Activation('relu'))


model.add(MaxPooling2D(pool_size=(2, 2))) #6X6X96 #RF=16X16
model.add(Dropout(0.25))

model.add(SeparableConv2D(192, kernel_size=(3,3), border_mode='same')) #6X6X192 #RF=24X24
model.add(BatchNormalization())
#model.add(Convolution2D(192, 3, 3, border_mode='same'))
model.add(Activation('relu'))

model.add(SeparableConv2D(192, kernel_size=(3,3))) #4X4X192 #RF=32X32
model.add(BatchNormalization())
#model.add(Convolution2D(192, 3, 3))
model.add(Activation('relu'))

model.add(MaxPooling2D(pool_size=(2, 2))) #2X2X192 #RF=36X36
model.add(Dropout(0.25))

model.add(SeparableConv2D(num_classes,kernel_size=(2,2))) #1X1X10 #RF=44X44
model.add(Flatten())

model.add(Activation('softmax'))

_Compiling the model_
model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

## Log for 50 epochs run

Epoch 1/50

Epoch 00001: LearningRateScheduler setting learning rate to 0.003.
781/781 [==============================] - 36s 46ms/step - loss: 1.3089 - acc: 0.5265 - val_loss: 1.3812 - val_acc: 0.5397
Epoch 2/50

Epoch 00002: LearningRateScheduler setting learning rate to 0.0022727273.
781/781 [==============================] - 32s 41ms/step - loss: 0.9588 - acc: 0.6608 - val_loss: 1.2795 - val_acc: 0.5696
Epoch 3/50

Epoch 00003: LearningRateScheduler setting learning rate to 0.0018292683.
781/781 [==============================] - 32s 41ms/step - loss: 0.8266 - acc: 0.7119 - val_loss: 1.6132 - val_acc: 0.5176
Epoch 4/50

Epoch 00004: LearningRateScheduler setting learning rate to 0.0015306122.
781/781 [==============================] - 33s 42ms/step - loss: 0.7485 - acc: 0.7393 - val_loss: 0.7139 - val_acc: 0.7512
Epoch 5/50

Epoch 00005: LearningRateScheduler setting learning rate to 0.0013157895.
781/781 [==============================] - 33s 42ms/step - loss: 0.6952 - acc: 0.7566 - val_loss: 0.6841 - val_acc: 0.7584
Epoch 6/50

Epoch 00006: LearningRateScheduler setting learning rate to 0.0011538462.
781/781 [==============================] - 33s 42ms/step - loss: 0.6606 - acc: 0.7691 - val_loss: 0.8573 - val_acc: 0.7130
Epoch 7/50

Epoch 00007: LearningRateScheduler setting learning rate to 0.0010273973.
781/781 [==============================] - 33s 42ms/step - loss: 0.6267 - acc: 0.7822 - val_loss: 0.6548 - val_acc: 0.7783
Epoch 8/50

Epoch 00008: LearningRateScheduler setting learning rate to 0.0009259259.
781/781 [==============================] - 33s 42ms/step - loss: 0.6053 - acc: 0.7877 - val_loss: 0.6429 - val_acc: 0.7812
Epoch 9/50

Epoch 00009: LearningRateScheduler setting learning rate to 0.0008426966.
781/781 [==============================] - 32s 41ms/step - loss: 0.5858 - acc: 0.7945 - val_loss: 0.6630 - val_acc: 0.7750
Epoch 10/50

Epoch 00010: LearningRateScheduler setting learning rate to 0.0007731959.
781/781 [==============================] - 32s 41ms/step - loss: 0.5662 - acc: 0.8036 - val_loss: 0.6060 - val_acc: 0.7885
Epoch 11/50

Epoch 00011: LearningRateScheduler setting learning rate to 0.0007142857.
781/781 [==============================] - 32s 41ms/step - loss: 0.5512 - acc: 0.8086 - val_loss: 0.5802 - val_acc: 0.8041
Epoch 12/50

Epoch 00012: LearningRateScheduler setting learning rate to 0.0006637168.
781/781 [==============================] - 32s 41ms/step - loss: 0.5368 - acc: 0.8106 - val_loss: 0.6029 - val_acc: 0.8036
Epoch 13/50

Epoch 00013: LearningRateScheduler setting learning rate to 0.0006198347.
781/781 [==============================] - 32s 41ms/step - loss: 0.5258 - acc: 0.8143 - val_loss: 0.5691 - val_acc: 0.8089
Epoch 14/50

Epoch 00014: LearningRateScheduler setting learning rate to 0.0005813953.
781/781 [==============================] - 32s 41ms/step - loss: 0.5160 - acc: 0.8187 - val_loss: 0.5965 - val_acc: 0.7991
Epoch 15/50

Epoch 00015: LearningRateScheduler setting learning rate to 0.0005474453.
781/781 [==============================] - 32s 41ms/step - loss: 0.4988 - acc: 0.8241 - val_loss: 0.5622 - val_acc: 0.8135
Epoch 16/50

Epoch 00016: LearningRateScheduler setting learning rate to 0.0005172414.
781/781 [==============================] - 32s 41ms/step - loss: 0.4988 - acc: 0.8261 - val_loss: 0.5567 - val_acc: 0.8098
Epoch 17/50

Epoch 00017: LearningRateScheduler setting learning rate to 0.0004901961.
781/781 [==============================] - 32s 41ms/step - loss: 0.4875 - acc: 0.8284 - val_loss: 0.6099 - val_acc: 0.7950
Epoch 18/50

Epoch 00018: LearningRateScheduler setting learning rate to 0.0004658385.
781/781 [==============================] - 32s 41ms/step - loss: 0.4800 - acc: 0.8315 - val_loss: 0.5440 - val_acc: 0.8159
Epoch 19/50

Epoch 00019: LearningRateScheduler setting learning rate to 0.000443787.
781/781 [==============================] - 32s 41ms/step - loss: 0.4753 - acc: 0.8325 - val_loss: 0.5296 - val_acc: 0.8224
Epoch 20/50

Epoch 00020: LearningRateScheduler setting learning rate to 0.0004237288.
781/781 [==============================] - 32s 41ms/step - loss: 0.4639 - acc: 0.8347 - val_loss: 0.5189 - val_acc: 0.8242
Epoch 21/50

Epoch 00021: LearningRateScheduler setting learning rate to 0.0004054054.
781/781 [==============================] - 32s 41ms/step - loss: 0.4612 - acc: 0.8379 - val_loss: 0.5327 - val_acc: 0.8194
Epoch 22/50

Epoch 00022: LearningRateScheduler setting learning rate to 0.000388601.
781/781 [==============================] - 32s 41ms/step - loss: 0.4513 - acc: 0.8408 - val_loss: 0.5483 - val_acc: 0.8212
Epoch 23/50

Epoch 00023: LearningRateScheduler setting learning rate to 0.0003731343.
781/781 [==============================] - 32s 41ms/step - loss: 0.4482 - acc: 0.8423 - val_loss: 0.5749 - val_acc: 0.8105
Epoch 24/50

Epoch 00024: LearningRateScheduler setting learning rate to 0.0003588517.
781/781 [==============================] - 32s 41ms/step - loss: 0.4378 - acc: 0.8456 - val_loss: 0.5347 - val_acc: 0.8233
Epoch 25/50

Epoch 00025: LearningRateScheduler setting learning rate to 0.0003456221.
781/781 [==============================] - 32s 41ms/step - loss: 0.4414 - acc: 0.8439 - val_loss: 0.5451 - val_acc: 0.8190
Epoch 26/50

Epoch 00026: LearningRateScheduler setting learning rate to 0.0003333333.
781/781 [==============================] - 32s 41ms/step - loss: 0.4363 - acc: 0.8462 - val_loss: 0.5101 - val_acc: 0.8329
Epoch 27/50

Epoch 00027: LearningRateScheduler setting learning rate to 0.0003218884.
781/781 [==============================] - 32s 41ms/step - loss: 0.4280 - acc: 0.8502 - val_loss: 0.5083 - val_acc: 0.8299
Epoch 28/50

Epoch 00028: LearningRateScheduler setting learning rate to 0.0003112033.
781/781 [==============================] - 32s 41ms/step - loss: 0.4246 - acc: 0.8502 - val_loss: 0.5142 - val_acc: 0.8296
Epoch 29/50

Epoch 00029: LearningRateScheduler setting learning rate to 0.0003012048.
781/781 [==============================] - 32s 41ms/step - loss: 0.4241 - acc: 0.8502 - val_loss: 0.5406 - val_acc: 0.8218
Epoch 30/50

Epoch 00030: LearningRateScheduler setting learning rate to 0.0002918288.
781/781 [==============================] - 32s 42ms/step - loss: 0.4176 - acc: 0.8528 - val_loss: 0.5096 - val_acc: 0.8302
Epoch 31/50

Epoch 00031: LearningRateScheduler setting learning rate to 0.0002830189.
781/781 [==============================] - 33s 42ms/step - loss: 0.4128 - acc: 0.8527 - val_loss: 0.5180 - val_acc: 0.8280
Epoch 32/50

Epoch 00032: LearningRateScheduler setting learning rate to 0.0002747253.
781/781 [==============================] - 33s 42ms/step - loss: 0.4152 - acc: 0.8525 - val_loss: 0.5070 - val_acc: 0.8321
Epoch 33/50

Epoch 00033: LearningRateScheduler setting learning rate to 0.0002669039.
781/781 [==============================] - 33s 42ms/step - loss: 0.4113 - acc: 0.8546 - val_loss: 0.4976 - val_acc: 0.8370
Epoch 34/50

Epoch 00034: LearningRateScheduler setting learning rate to 0.0002595156.
781/781 [==============================] - 32s 41ms/step - loss: 0.4073 - acc: 0.8573 - val_loss: 0.5196 - val_acc: 0.8297
Epoch 35/50

Epoch 00035: LearningRateScheduler setting learning rate to 0.0002525253.
781/781 [==============================] - 32s 41ms/step - loss: 0.4113 - acc: 0.8531 - val_loss: 0.5136 - val_acc: 0.8298
Epoch 36/50

Epoch 00036: LearningRateScheduler setting learning rate to 0.0002459016.
781/781 [==============================] - 32s 41ms/step - loss: 0.4016 - acc: 0.8588 - val_loss: 0.5104 - val_acc: 0.8315
Epoch 37/50

Epoch 00037: LearningRateScheduler setting learning rate to 0.0002396166.
781/781 [==============================] - 32s 41ms/step - loss: 0.3973 - acc: 0.8593 - val_loss: 0.4971 - val_acc: 0.8354
Epoch 38/50

Epoch 00038: LearningRateScheduler setting learning rate to 0.0002336449.
781/781 [==============================] - 32s 41ms/step - loss: 0.3982 - acc: 0.8592 - val_loss: 0.5059 - val_acc: 0.8369
Epoch 39/50

Epoch 00039: LearningRateScheduler setting learning rate to 0.0002279635.
781/781 [==============================] - 32s 41ms/step - loss: 0.3971 - acc: 0.8592 - val_loss: 0.4958 - val_acc: 0.8350
Epoch 40/50

Epoch 00040: LearningRateScheduler setting learning rate to 0.0002225519.
781/781 [==============================] - 32s 41ms/step - loss: 0.3913 - acc: 0.8610 - val_loss: 0.5072 - val_acc: 0.8338
Epoch 41/50

Epoch 00041: LearningRateScheduler setting learning rate to 0.0002173913.
781/781 [==============================] - 32s 42ms/step - loss: 0.3904 - acc: 0.8607 - val_loss: 0.5124 - val_acc: 0.8307
Epoch 42/50

Epoch 00042: LearningRateScheduler setting learning rate to 0.0002124646.
781/781 [==============================] - 32s 41ms/step - loss: 0.3845 - acc: 0.8630 - val_loss: 0.5087 - val_acc: 0.8319
Epoch 43/50

Epoch 00043: LearningRateScheduler setting learning rate to 0.0002077562.
781/781 [==============================] - 32s 41ms/step - loss: 0.3854 - acc: 0.8645 - val_loss: 0.5041 - val_acc: 0.8373
Epoch 44/50

Epoch 00044: LearningRateScheduler setting learning rate to 0.000203252.
781/781 [==============================] - 32s 41ms/step - loss: 0.3815 - acc: 0.8657 - val_loss: 0.5036 - val_acc: 0.8357
Epoch 45/50

Epoch 00045: LearningRateScheduler setting learning rate to 0.000198939.
781/781 [==============================] - 32s 41ms/step - loss: 0.3802 - acc: 0.8641 - val_loss: 0.5104 - val_acc: 0.8329
Epoch 46/50

Epoch 00046: LearningRateScheduler setting learning rate to 0.0001948052.
781/781 [==============================] - 32s 42ms/step - loss: 0.3795 - acc: 0.8653 - val_loss: 0.5029 - val_acc: 0.8355
Epoch 47/50

Epoch 00047: LearningRateScheduler setting learning rate to 0.0001908397.
781/781 [==============================] - 32s 41ms/step - loss: 0.3829 - acc: 0.8641 - val_loss: 0.5185 - val_acc: 0.8309
Epoch 48/50

Epoch 00048: LearningRateScheduler setting learning rate to 0.0001870324.
781/781 [==============================] - 32s 41ms/step - loss: 0.3749 - acc: 0.8671 - val_loss: 0.4966 - val_acc: 0.8345
Epoch 49/50

Epoch 00049: LearningRateScheduler setting learning rate to 0.0001833741.
781/781 [==============================] - 32s 41ms/step - loss: 0.3767 - acc: 0.8658 - val_loss: 0.5081 - val_acc: 0.8346
Epoch 50/50

Epoch 00050: LearningRateScheduler setting learning rate to 0.0001798561.
781/781 [==============================] - 32s 41ms/step - loss: 0.3706 - acc: 0.8689 - val_loss: 0.4980 - val_acc: 0.8379
Model took 1619.28 seconds to train

Accuracy on test data is: 83.79(greater than 83)

Parameters used: 57,568(less than 100,000)
