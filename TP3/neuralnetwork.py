"""
Machine Learning: Handwritten Number Recognition
Dataset: MNIST Data
Inspired from: http://deeplearning.net/tutorial/mlp.html

Eric Morissette (1631103)
Sacha Licatese-Roussel (1635849)
"""

# Draw
import Tkinter as tk

# Open/Parse the MNIST Data
import gzip
import pickle

# Show an image
import matplotlib.cm as cm
import matplotlib.pyplot as plt

# Matrix computation
import numpy
import theano
import theano.tensor as T

import sys
def customPrint(text, newLine = True):
    sys.stdout.write(text + ('\n' if newLine else ''))

def showImage(img):
    plt.imshow(img.reshape((28, 28)), cmap = cm.Greys_r)
    plt.show()

def loadDataset():
    customPrint('Loading data...', False)
    with gzip.open('mnist.pkl.gz', 'rb') as f:
        #Python3 needs "encoding='latin1'"
        try:
            train, validate, test = pickle.load(f, encoding='latin1')
        except:
            train, validate, test = pickle.load(f)

        def shared_dataset(data_xy, borrow=True):

            data_x, data_y = data_xy
            shared_x = theano.shared(numpy.asarray(data_x,
                            dtype=theano.config.floatX), borrow=True)
            shared_y = theano.shared(numpy.asarray(data_y,
                            dtype=theano.config.floatX), borrow=True)
            return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y   = shared_dataset(test)
    valid_set_x, valid_set_y = shared_dataset(validate)
    train_set_x, train_set_y = shared_dataset(train)

    customPrint('Done', True)

    return [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]

class HiddenLayer(object):
    def __init__(self, input, n_in, n_out, W = None, b = None):
        """
        Hidden Layer of our Neural Network

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        input: tensor of shape (n_examples, n_in)

        n_in: input's dimension

        n_out: output's dimension

        W: weight matrix of size n_in*n_out

        b: bias vector of size n_out*1.
        """
        self.input = input

        # If no W is provided, generate one using random values
        if W is None:
            lowHigh = numpy.sqrt(6. / (n_in + n_out))
            W_values = numpy.asarray(
                numpy.random.RandomState(1234).uniform(
                    low = -lowHigh, high = lowHigh,
                    size = (n_in, n_out)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value = W_values, name = 'W', borrow = True)
        self.W = W

        # If no B is provided, generate one
        if b is None:
            b = theano.shared(value = numpy.zeros((n_out,),
                    dtype = theano.config.floatX), name = 'b', borrow = True)
        self.b = b

        self.output = T.tanh(T.dot(input, self.W) + self.b)

        # Combine W and B
        self.params = [self.W, self.b]

class OutputLayer(object):
    def __init__(self, input, n_in, n_out):
        """
        Hidden Layer of our Neural Network

        input: theano tensor.TensorType

        n_in: inputs

        n_out: number of outputs

        """
        self.input = input

        # Initialize W as a matrix of zeros
        self.W = theano.shared( value = numpy.zeros((n_in, n_out),
                    dtype = theano.config.floatX ), name = 'W', borrow = True)

        # Initialize W as a vector of zeros
        self.b = theano.shared( value = numpy.zeros((n_out,),
                    dtype = theano.config.floatX ), name = 'b', borrow = True)

        # Probability of Y given X
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # Combine W and B
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        return T.mean(T.neq(self.y_pred, y))

    def check(self):
        return self.p_y_given_x

class NeuralNetwork(object):
    def __init__(self, input, n_in, n_hidden, n_out):
        """
        Neural Network

        input: theano tensor

        n_in: number of input units

        n_hidden: number of hidden units

        n_out: number of output units

        """
        self.input = input

        # We will use only one hidden layer:
        # Input  =  28x28 = 784
        # Hidden = 500x 1 = 500
        # Output =  10x 1 =  10
        self.hiddenLayer = HiddenLayer(input=input,
                                       n_in=n_in, n_out=n_hidden)

        # The output layer is linked with the hidden layer
        self.outputLayer = OutputLayer( input=self.hiddenLayer.output,
                                       n_in=n_hidden, n_out=n_out)

        # L1 Regularization
        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.outputLayer.W).sum()

        # L2 Regularization (Squared)
        self.L2 = (self.hiddenLayer.W ** 2).sum() + (self.outputLayer.W ** 2).sum()

        # negative log likelihood of the NN, computed in the output layer
        self.negative_log_likelihood = self.outputLayer.negative_log_likelihood

        # same holds for the function computing the number of errors
        self.errors = self.outputLayer.errors

        # the parameters of the model are the parameters of the two layer it is
        # made out of
        self.params = self.hiddenLayer.params + self.outputLayer.params

        self.check = self.outputLayer.check

def defaultNeuralNetwork():

    # Default Parameters
    learning_rate   = 0.01
    L1_coeff        = 0.00
    L2_coeff        = 0.0001
    n_epochs        = 1
    batch_size      = 100
    n_hidden        = 500

    # Load the MNIST dataset
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = loadDataset()

    # Mini-batches
    n_train_batches = train_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches  = test_x.get_value(borrow=True).shape[0] // batch_size

    # Build the Neural Network's layers
    customPrint('Building...', False)

    index = T.lscalar() # index to a mini-batch
    x = T.matrix('x')   # Input Matrix, 28x28
    y = T.ivector('y')  # "Answer" Vector, 10x1

    # Create the Neural Network
    classifier = NeuralNetwork(input = x, n_in = 28 * 28,
        n_hidden = n_hidden, n_out = 10)

    # Define the cost function, which is modified by the L1L2 Regularization
    cost = ( classifier.negative_log_likelihood(y)
        + L1_coeff * classifier.L1 + L2_coeff * classifier.L2 )

    # Compile Theano functions to test/validate the model
    validate_model = theano.function(
        inputs = [index], outputs = classifier.errors(y),
        givens = {
            x: valid_x[index * batch_size:(index + 1) * batch_size],
            y: valid_y[index * batch_size:(index + 1) * batch_size]
        }
    )
    test_model = theano.function(
        inputs = [index], outputs = classifier.errors(y),
        givens = {
            x: test_x[index * batch_size:(index + 1) * batch_size],
            y: test_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # User input test
    img = T.matrix('img')
    userInputTestTheano = theano.function(
        inputs = [img], outputs = classifier.check(),
        givens = {
            x: img
        }
    )
    def userInputTest(testImage):
        return userInputTestTheano(testImg)[0]

    # Gradient Params
    gradParams = [T.grad(cost, param) for param in classifier.params]

    # Merge the lists together, if there are multiple
    updates = [ (param, param - learning_rate * gradParam)
        for param, gradParam in zip(classifier.params, gradParams) ]

    # Compile the Theano function responsible for updating the Neural Network
    train_model = theano.function(
        inputs = [index], outputs = cost, updates = updates,
        givens = {
            x: train_x[index * batch_size: (index + 1) * batch_size],
            y: train_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    # Building is done
    customPrint('Done', True)

    # Start the training
    customPrint('Training...', False)

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.

    for epoch in range(n_epochs):
        for minibatch_index in range(n_train_batches):
            train_model(minibatch_index)
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % n_train_batches == 0:
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                customPrint('Validation Error Percentage: ' + str(this_validation_loss * 100.)
                    + ' obtained on iteration ' + str(minibatch_index + 1)
                    + ' at epoch ' + str(epoch), True)

                # new best validation
                if this_validation_loss < best_validation_loss:
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test on the test set
                    test_losses = [test_model(i) for i
                        in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    customPrint('New Best Validation at iteration '
                        + str(minibatch_index + 1) + ' at epoch ' + str(epoch), True)
                    customPrint('New Test Error Percentage: ' + str(test_score * 100.), True)

    # Training is done
    customPrint('Done', True)

    customPrint('Best Validation: ' + str(best_validation_loss * 100.), True)
    customPrint(' -  Test Result: ' + str(test_score * 100.), True)
    customPrint(' - On iteration: ' + str(best_iter + 1), True)

    root=tk.Tk()
    root.wm_geometry("%dx%d+%d+%d" % (400, 400, 10, 10))
    root.config(bg='white')
    ImageGenerator(root,10,10)
    root.mainloop()



class ImageGenerator:
    def __init__(self,parent,posx,posy,*kwargs):
        self.parent = parent
        self.posx = posx
        self.posy = posy
        self.sizex = 280
        self.sizey = 280
        self.b1 = "up"
        self.xold = None
        self.yold = None 
        self.drawing_area=tk.Canvas(self.parent,width=self.sizex,height=self.sizey)
        self.drawing_area.place(x=self.posx,y=self.posy)
        self.drawing_area.bind("<Motion>", self.motion)
        self.drawing_area.bind("<ButtonPress-1>", self.b1down)
        self.drawing_area.bind("<ButtonRelease-1>", self.b1up)
        self.button=tk.Button(self.parent,text="Submit",width=10,bg='white',command=self.submit)
        self.button.place(x=self.sizex/7,y=self.sizey+20)
        self.button1=tk.Button(self.parent,text="Clear",width=10,bg='white',command=self.clear)
        self.button1.place(x=(self.sizex/7)+80,y=self.sizey+50)

        self.pixelArray = numpy.ones((self.sizex, self.sizey))

    def submit(self):
        #make putin some poutine
        tempArray = numpy.zeros((1, 28*28))
        for i in range(0, self.sizex):
            for j in range(0, self.sizey):
                tempArray[0, ((j // 10) + (28 * (i // 10)))] += (1 - self.pixelArray[i, j])

        tempArray /= 100.

        ans = userInputTest(testImg)
        for i in range(len(ans)):
            print('Chance of ' + str(i) + ': ' + str(ans[i]))

    def clear(self):
        self.drawing_area.delete("all")
        self.pixelArray = numpy.ones((self.sizex, self.sizey))

    def b1down(self,event):
        self.b1 = "down"

    def b1up(self,event):
        self.b1 = "up"
        self.xold = None
        self.yold = None

    def motion(self,event):
        if self.b1 == "down":
            if self.xold is not None and self.yold is not None:
                event.widget.create_line(self.xold,self.yold,event.x,event.y,smooth=True,width=7,fill='black')
                if (event.x >= 1 and event.x < self.sizex - 1 and event.y >= 1 and event.y < self.sizey - 1):
                    for i in range(-3, 4):
                        for j in range(-3, 4):
                            self.pixelArray[event.y + i, event.x + j] = 0
        self.xold = event.x
        self.yold = event.y

if __name__ == '__main__':
    defaultNeuralNetwork()


