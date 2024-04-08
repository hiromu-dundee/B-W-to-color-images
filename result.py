#PS C:\Users\owner\.vscode\code\Python\ml\programs> 内で実行する

import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
import cv2
import glob
import matplotlib.pyplot as plt
from icrawler.builtin import BingImageCrawler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, UpSampling2D, InputLayer, MaxPool2D, Dropout, BatchNormalization, Activation
from tensorflow.python.keras.optimizers import adam_v2

# カラー画像と白黒画像のペアを用意する関数
def prepare_data(img_paths, img_size=(96, 96)):
    color_imgs = []
    gray_imgs = []

    for img_path in img_paths:
        # カラー画像
        color_img = cv2.imread(img_path)
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)
        color_img = cv2.resize(color_img, img_size)

        # 白黒画像
        
        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        gray_img = cv2.resize(gray_img, img_size)
        #gray_img = np.expand_dims(gray_img, axis=-1)  # チャンネル次元を追加
        

        '''
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        neiborhood = np.ones((5, 5), dtype=np.uint8)
        dilated = cv2.dilate(img, neiborhood, iterations=5)

        diff = cv2.absdiff(dilated, img)

        #5. 白黒反転
        gray_img = 255 - diff

        gray_img = cv2.resize(gray_img, img_size)
        gray_img = np.expand_dims(gray_img, axis=-1)  # チャンネル次元を追加
        '''

        color_imgs.append(color_img)
        gray_imgs.append(gray_img)

    gray_imgs = np.array(gray_imgs) / 255.0
    color_imgs = np.array(color_imgs) / 255.0

    #return np.array(color_imgs), np.array(gray_imgs)
    return color_imgs, gray_imgs

# データセットのパス
dataset_paths = glob.glob('Python/ml/image3/slime3_redblue/*.png') 

# カラー画像と白黒画像のペアを用意
color_images, gray_images = prepare_data(dataset_paths)

# データを訓練データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(gray_images, color_images, test_size=0.2, random_state=42)

# モデルの定義
model = Sequential()
model.add(InputLayer(shape=(96, 96, 1)))

model.add(Conv2D(3, (3, 3), padding='same', strides=1))
model.add(BatchNormalization(momentum=0.9, epsilon=1e-5))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), padding='same', strides=1))
model.add(BatchNormalization(momentum=0.9, epsilon=1e-5))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), padding='same', strides=1))
model.add(BatchNormalization(momentum=0.9, epsilon=1e-5))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv2D(256, (3, 3), padding='same', strides=1))
model.add(BatchNormalization(momentum=0.9, epsilon=1e-5))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv2D(128, (3, 3), padding='same', strides=1))
model.add(BatchNormalization(momentum=0.9, epsilon=1e-5))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3), padding='same', strides=1))
model.add(BatchNormalization(momentum=0.9, epsilon=1e-5))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Conv2D(3, (3, 3), padding='same', strides=1))  # Sigmoid から Linear に変更
model.add(BatchNormalization(momentum=0.9, epsilon=1e-5))
model.add(Activation('sigmoid'))

# モデルのコンパイル
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002)
model.compile(optimizer=optimizer, loss='mae', metrics=['accuracy'])


print(model.summary())

# モデルの学習
history = model.fit(X_train, y_train, epochs=30, batch_size=100, validation_data=(X_test, y_test))

# ある白黒データに対する予測
input_gray_image = X_test[0].reshape(1, 96, 96, 1)
predicted_color_image = model.predict(input_gray_image)

# 結果の表示
plt.subplot(1, 3, 1)
plt.title('Input Grayscale Image')
plt.imshow(X_test[0].reshape(96, 96), cmap='gray')  # グレースケール画像なので256x256に修正

plt.subplot(1, 3, 2)
plt.title('Input RGB Image')
plt.imshow(y_test[0])

plt.subplot(1, 3, 3)
plt.title('Predicted Color Image')
plt.imshow((predicted_color_image[0] * 255).astype(np.uint8))

plt.show()

# 学習過程の損失と精度の情報を取得
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Epochごとのグラフを描画
epochs = range(1, len(train_loss) + 1)

# 損失のグラフ
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# 精度のグラフ
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

