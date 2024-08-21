# Imports

import numpy as np #importing numpy and aliasing it as np for convenince
from numpy import random #importing the submodule random from numpy
from matplotlib import pyplot as plt #importing the submodule pyplot from matplotlib and aliasing that as plt


#this line is needed to display images within a Jupyter Notebok
%matplotlib inline 

##### Drawing function

def view_samples(samples, m, n):
    fig, axes = plt.subplots(figsize=(10, 10), nrows=m, ncols=n, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(1-img.reshape((2,2)), cmap='Greys_r')  
    return fig, axes

# Examples of faces
faces = [np.array([1,0,0,1]),
         np.array([0.9,0.1,0.2,0.8]),
         np.array([0.9,0.2,0.1,0.8]),
         np.array([0.8,0.1,0.2,0.9]),
         np.array([0.8,0.2,0.1,0.9])]


faces_view = view_samples(faces, 4, 3)

# Examples of noisy images
noise = [np.random.randn(2,2) for i in range(25)]
def generate_random_image():
    return [np.random.random(), np.random.random(), np.random.random(), np.random.random()]


noise_view = view_samples(noise, 5,5)

# The sigmoid activation function
def sigmoid(x):
    return np.exp(x)/(1.0+np.exp(x))

sigmoid(1.5)

class Discriminator():
    def __init__(self):
        #self.weights = np.array([0.0 for i in range(4)])
        #self.bias = 0.0
        self.weights = np.array([np.random.normal() for i in range(4)])
        self.bias = np.random.normal()
    
    def forward(self, x):
        # Forward pass
        return sigmoid(np.dot(x, self.weights) + self.bias)
    
    #error from real image
    def error_from_image(self, image):
        prediction = self.forward(image)
        # We want the prediction to be 1, so the error is -log(prediction)
        return -np.log(prediction)
    
    #derivative from real image
    def derivatives_from_image(self, image):
        prediction = self.forward(image)
        derivatives_weights = -image * (1-prediction)
        derivative_bias = -(1-prediction)
        return derivatives_weights, derivative_bias
    
    #weights update from real image
    def update_from_image(self, x):
        ders = self.derivatives_from_image(x)
        self.weights -= learning_rate * ders[0]
        self.bias -= learning_rate * ders[1]

    #error from fake image
    def error_from_noise(self, noise):
        prediction = self.forward(noise)
        # We want the prediction to be 0, so the error is -log(1-prediction)
        return -np.log(1-prediction)
    
    #derivative from fake image
    def derivatives_from_noise(self, noise):
        prediction = self.forward(noise)
        derivatives_weights = noise * prediction
        derivative_bias = prediction
        return derivatives_weights, derivative_bias
    
    #weights update from fake image
    def update_from_noise(self, noise):
        ders = self.derivatives_from_noise(noise)
        self.weights -= learning_rate * ders[0]
        self.bias -= learning_rate * ders[1]

  class Generator():
    #generate random number
    def __init__(self):
        self.weights = np.array([np.random.normal() for i in range(4)])
        self.biases = np.array([np.random.normal() for i in range(4)])

    def forward(self, z):
        # Forward pass
        return sigmoid(z * self.weights + self.biases)

    def error(self, z, discriminator):
        x = self.forward(z)
        # We want the prediction to be 0, so the error is -log(1-prediction)
        y = discriminator.forward(x)
        return -np.log(y)

    def derivatives(self, z, discriminator):
        discriminator_weights = discriminator.weights
        discriminator_bias = discriminator.bias
        x = self.forward(z)
        y = discriminator.forward(x)
        factor = -(1-y) * discriminator_weights * x *(1-x)
        derivatives_weights = factor * z
        derivative_bias = factor
        return derivatives_weights, derivative_bias

    def update(self, z, discriminator):
        error_before = self.error(z, discriminator)
        ders = self.derivatives(z, discriminator)
        self.weights -= learning_rate * ders[0]
        self.biases -= learning_rate * ders[1]
        error_after = self.error(z, discriminator)



# Set random seed
np.random.seed(42)

# Hyperparameters
learning_rate = 0.07
epochs = 10000

# defining the discriminator and generator
D = Discriminator()
G = Generator()

# For the error plot
errors_discriminator = []
errors_generator = []

#Repeat the following instructions for the number in epochs
for epoch in range(epochs):
    
    # For every real image (generated in part 2)
    for face in faces:
        
        # Update the discriminator weights from the real face
        D.update_from_image(face)
    
        z = random.rand()

        # Calculate the discriminator error
        errors_discriminator.append(sum(D.error_from_image(face) + D.error_from_noise(z)))
        
        # Calculate the generator error
        errors_generator.append(G.error(z, D))
        
        noise = G.forward(z)
        
        # Update the discriminator weights from the fake face
        D.update_from_noise(noise)
    
        # Update the generator weights from the fake face
        G.update(z, D)



#Plotting the errors in the generator
plt.plot(errors_generator)
#Setting plot title
plt.title("Generator error function")
#Setting legend
plt.legend("g")
plt.show()


#Plotting the errors in the discriminator
plt.plot(errors_discriminator)
#Setting plot title
plt.title("Discriminator error function")
#Setting legend
plt.legend('d')


