from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
from tensorflow.keras.applications import ResNet50V2

num_class = 2
epoch = 6
image_width = 100
image_height = 100
batch_size = 64
input_shape = (image_width, image_height, 3)
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
tensorboard = TensorBoard()
# load data. download dataset in https://centennialcollegeedu-my.sharepoint.com/:u:/g/personal/jtan73_my_centennialcollege_ca/EURYk12qubVEi936zWSnEMQBnLyGsNTbqN4nBOeBmNG1hA?e=hO0E9s
data = ImageDataGenerator(rescale=1./255)
train_generator = data.flow_from_directory('dataset/train', target_size=(image_width, image_height), shuffle=True, batch_size=batch_size)
validation_generator = data.flow_from_directory('dataset/validation', target_size=(image_width, image_height), shuffle=True, batch_size=batch_size)

# modeling
model = ResNet50V2(weights=None, input_shape=input_shape, classes=num_class, classifier_activation='sigmoid')
model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])
model.fit(train_generator, validation_data=validation_generator, epochs=epoch, callbacks=[early_stopping, tensorboard], workers=4)
model.save('model')