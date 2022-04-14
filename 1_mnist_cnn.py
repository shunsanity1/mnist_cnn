import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt


n_classes = 10
img_rows, img_cols = 28, 28


def display_sample(X_train, y_train):
    global n_classes, img_rows, img_cols

    pos = 1
    for targetClass in range(n_classes):
        targetIdx = []
        # クラスclassIDの画像のインデックスリストを取得
        for i in range(len(y_train)):
            if y_train[i] == targetClass:
                targetIdx.append(i)

        # 各クラスからランダムに選んだ最初の10個の画像を描画
        np.random.shuffle(targetIdx)
        for idx in targetIdx[:10]:
            img = cv2.cvtColor(X_train[idx].reshape(img_rows, img_cols), cv2.COLOR_GRAY2BGR)
            plt.subplot(10, 10, pos)
            plt.imshow(img)
            plt.axis('off')
            pos += 1

    plt.show()


def LeNet5():
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    from keras.layers import Conv2D, MaxPooling2D, Activation

    global n_classes, img_rows, img_cols

    # CNNを構築
    model = Sequential()

    model.add(Conv2D(6, (5, 5), padding='same', input_shape=(img_rows, img_cols, 1)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    model.add(Conv2D(16, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #
    model.add(Flatten())
    model.add(Dense(120))
    model.add(Activation('relu'))
    model.add(Dense(84))
    model.add(Activation('relu'))
    #
    model.add(Dense(n_classes))
    model.add(Activation('softmax'))

    return model


def main():
    from keras.callbacks import EarlyStopping, ModelCheckpoint
    from keras.datasets import mnist
    from keras.utils import to_categorical

    global n_classes, img_rows, img_cols

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

    # 画素値を0-1に変換
    X_train = X_train.astype('float32') / 255.0
    X_test = X_test.astype('float32') / 255.0

    # クラスラベル（0-9）をone-hotエンコーディング形式に変換
    Y_train = to_categorical(y_train, n_classes)
    Y_test = to_categorical(y_test, n_classes)

    model = LeNet5()
    #model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # コールバック関数の設定
    es_cb = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='auto')
    cp_cb = ModelCheckpoint(filepath = './model.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')

    # 訓練
    history = model.fit(X_train, Y_train,
                        batch_size=32,
                        epochs=100,
                        verbose=1,
                        validation_split=0.1,
                        callbacks=[cp_cb, es_cb]
                        )

    # 評価
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import confusion_matrix
    import numpy as np

    # テストデータに対する精度
    pred_test = model.predict(X_test)
    pred_test = np.argmax(pred_test, axis=1)

    Y_test = np.argmax(Y_test, axis=1)

    accuracy_test = accuracy_score(Y_test, pred_test)
    print('テストデータに対する正解率： %.4f' % accuracy_test)
    print(confusion_matrix(Y_test, pred_test))

if __name__ == '__main__':
    main()
