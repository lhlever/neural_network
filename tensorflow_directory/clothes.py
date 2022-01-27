import tensorflow as tf
from keras import regularizers
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__)
# 1.导入数据集
fashion_mnis=keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_mnis.load_data()
print(train_images[0])
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
# 2.预处理数据
'''
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()
'''
# 3.归一化
train_images=train_images/255.0 #广播
test_images=test_images/255.0
# 4.显示前25个样本
'''
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i],cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()
'''
# 5.构建模型
#  5.1设置层
model=keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),#把28*28展平为784*1
    keras.layers.Dense(256,activation='relu',kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64,activation='relu',kernel_regularizer=regularizers.l2(0.001)),
    keras.layers.Dense(10)
])
#  5.2编译模型
# 损失函数 - 用于测量模型在训练期间的准确率。您会希望最小化此函数，以便将模型“引导”到正确的方向上。
# 优化器 - 决定模型如何根据其看到的数据和自身的损失函数进行更新。
# 指标 - 用于监控训练和测试步骤。以下示例使用了准确率，即被正确分类的图像的比率。
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy']
              )
#  5.3训练模型
# epochs迭代次数
model.fit(train_images,train_labels,epochs=15)
test_loss,test_acc=model.evaluate(test_images,test_labels,verbose=2)
print('\nTest accuracy:',test_acc)
# 6.