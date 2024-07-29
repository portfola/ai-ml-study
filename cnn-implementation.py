# Python Implementation of CNNs

# Import the necessary libraries.
import tensorflow as tf
from tensorflow.keras import layers, models

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Preprocess the data
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Build the CNN model
# The first step is to create a sequential model. This is a common approach in Keras for building neural networks with a linear stack of layers. 
model = models.Sequential()

# Next, the Conv2D layer is added. It is a fundamental building block in CNNs and plays a crucial role in enabling the network to learn features from input data.
# The Conv2D layer has several parameters that can be adjusted, such as the number of filters, the size of the filters, the step size of the convolution operation, and the padding method.
# These parameters affect how the convolution operation is applied and can impact the network's ability to learn and generalize from the data.
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))

# Next, a max pooling layer with a pool size of (2, 2) is added. This down samples the feature maps to reduce computation.
model.add(layers.MaxPooling2D((2, 2)))

# The high-level features are learned by using a dense layer with 64 units and a ReLU activation function.
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

# The MNIST dataset helps in classification of digits from 0 to 9.
# This classification is accomplished by using a dense layer with ten units and the softmax activation function for multi-class classification.
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
# The compile() method is called on the model object, where the optimizer, loss function, and metrics are specified. 
# This step creates the computation graph for the model and prepares it for training. 
# The optimizer used in this case is an “adam” optimizer, and the loss function is a "sparse_categorical_crossentropy." 
# The accuracy is the metric, which is computed here.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
# The model is trained using the model.fit method. The parameters in this case are the training data, the test data, and the number of epochs.
model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

# Evaluate the model
# Model evaluation is a crucial step in the machine learning workflow, as it provides insights into how well the model generalizes 
# to unseen data and helps assess its suitability for real-world applications. Here, the test_loss and test_acc values are computed 
# using the model.evaluate function on the test dataset.
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f'Test accuracy: {test_acc}')
