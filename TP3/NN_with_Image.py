"""
Machine Learning: Handwritten Number Recognition
Dataset: MNIST Data
Inspired from: http://deeplearning.net/tutorial/mlp.html

Eric Morissette (1631103)
Sacha Licatese-Roussel (1635849)
"""

# Draw
import Tkinter as tk
import tkMessageBox
import math

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

class ImageGenerator:
    def __init__(self,posx,posy,*kwargs):
        self.posx = posx
        self.posy = posy
        self.sizex = 280
        self.sizey = 280
        self.b1 = "up"
        self.xold = None
        self.yold = None 

        self.pixelArray = numpy.ones((self.sizex, self.sizey))

        # NEURAL NETWORK
        # Default Parameters
        self.learning_rate   = 0.01
        self.L1_coeff        = 0.00
        self.L2_coeff        = 0.0001
        self.n_epochs        = input("Entrez le nombre d'epoch a executer : ")
        self.batch_size      = 100

        # Network size
        self.n_inputLayer    = 28*28
        self.n_hiddenLayer   = 500
        self.n_outputLayer   = 10

        self.nn_index = T.lscalar() # index to a mini-batch
        self.nn_x = T.matrix('x')   # Input Matrix, 28x28
        self.nn_y = T.ivector('y')  # "Answer" Vector, 10x1
        self.nn_img = T.matrix('img')

        # Load the MNIST dataset
        (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = loadDataset()

        # Mini-batches
        self.n_train_batches = train_x.get_value(borrow=True).shape[0] // self.batch_size
        self.n_valid_batches = valid_x.get_value(borrow=True).shape[0] // self.batch_size
        self.n_test_batches  = test_x.get_value(borrow=True).shape[0] // self.batch_size

        # Create the Neural Network
        self.classifier = NeuralNetwork(input = self.nn_x, n_in = self.n_inputLayer,
            n_hidden = self.n_hiddenLayer, n_out = self.n_outputLayer)

        # Define the cost function, which is modified by the L1L2 Regularization
        cost = ( self.classifier.negative_log_likelihood(self.nn_y)
            + self.L1_coeff * self.classifier.L1 + self.L2_coeff * self.classifier.L2 )

        # Gradient Params
        gradParams = [T.grad(cost, param) for param in self.classifier.params]

        # Merge the lists together, if there are multiple
        updates = [ (param, param - self.learning_rate * gradParam)
            for param, gradParam in zip(self.classifier.params, gradParams) ]

        # Compile the Theano function responsible for updating the Neural Network
        train_model = theano.function(
            inputs = [self.nn_index], outputs = cost, updates = updates,
            givens = {
                self.nn_x: train_x[self.nn_index * self.batch_size: (self.nn_index + 1) * self.batch_size],
                self.nn_y: train_y[self.nn_index * self.batch_size: (self.nn_index + 1) * self.batch_size]
            }
        )

        # Compile Theano functions to test/validate the model
        validate_model = theano.function(
            inputs = [self.nn_index], outputs = self.classifier.errors(self.nn_y),
            givens = {
                self.nn_x: valid_x[self.nn_index * self.batch_size:(self.nn_index + 1) * self.batch_size],
                self.nn_y: valid_y[self.nn_index * self.batch_size:(self.nn_index + 1) * self.batch_size]
            }
        )

        test_model = theano.function(
            inputs = [self.nn_index], outputs = self.classifier.errors(self.nn_y),
            givens = {
                self.nn_x: test_x[self.nn_index * self.batch_size:(self.nn_index + 1) * self.batch_size],
                self.nn_y: test_y[self.nn_index * self.batch_size:(self.nn_index + 1) * self.batch_size]
            }
        )

        # User input test
        self.userInputTestTheano = theano.function(
            inputs = [self.nn_img], outputs = self.classifier.check(),
            givens = {
                self.nn_x: self.nn_img
            }
        )

        best_validation_loss = numpy.inf
        best_iter = 0
        test_score = 0.

        for epoch in range(self.n_epochs):
            for minibatch_index in range(self.n_train_batches):
                train_model(minibatch_index)
                iter = (epoch - 1) * self.n_train_batches + minibatch_index

                if (iter + 1) % self.n_train_batches == 0:
                    validation_losses = [validate_model(i) for i
                                         in range(self.n_valid_batches)]
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
                            in range(self.n_test_batches)]
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
        root.wm_geometry("%dx%d+%d+%d" % (300, 370, 10, 10))
        root.config(bg='white')
        root.wm_title("INF4215 - TP3")
        self.drawing_area=tk.Canvas(root, width=self.sizex,height=self.sizey)
        self.drawing_area.place(x=self.posx,y=self.posy)
        self.drawing_area.bind("<Motion>", self.motion)
        self.drawing_area.bind("<ButtonPress-1>", self.b1down)
        self.drawing_area.bind("<ButtonRelease-1>", self.b1up)
        self.button=tk.Button(root,text="Submit",width=10,bg='white',command=self.submit)
        self.button.place(x=self.sizex/3,y=self.sizey+20)
        self.button1=tk.Button(root,text="Clear",width=10,bg='white',command=self.clear)
        self.button1.place(x=(self.sizex/3),y=self.sizey+50)
        root.mainloop()

    def submit(self):
        tempArray = numpy.zeros((1, 28*28))

        for i in range(0, self.sizex):
            for j in range(0, self.sizey):
                tempArray[0, ((j // 10) + (28 * (i // 10)))] += (1 - self.pixelArray[i, j])

        tempArray /= 100.

        ans = self.userInputTest(tempArray)
        currMax = -1
        currMaxProb = 0
        for i in range(len(ans)):
            print('Chance of ' + str(i) + ': ' + str(ans[i]))
            if (currMaxProb < ans[i]):
                currMax = i
                currMaxProb = ans[i]
        print('\n')
        tkMessageBox.showinfo("Resultat", ("Voici le nombre le plus probable: " + str(currMax)))

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
                largeurTrait = 11
                largeurTraitMoitie = math.floor(largeurTrait / 2)
                #event.widget.create_line(self.xold,self.yold,event.x,event.y,smooth=True,width=largeurTrait,fill='black')
                x0 = int(event.x - largeurTraitMoitie)
                y0 = int(event.y - largeurTraitMoitie)
                x1 = int(event.x + largeurTraitMoitie)
                y1 = int(event.y + largeurTraitMoitie)
                event.widget.create_oval(x0, y0, x1, y1,fill='black')
                if (event.x >= largeurTraitMoitie and event.x < self.sizex - largeurTraitMoitie and event.y >= largeurTraitMoitie and event.y < self.sizey - largeurTraitMoitie):
                    for i in range(0, largeurTrait):
                        for j in range(0, largeurTrait):
                            self.pixelArray[event.y + i - largeurTraitMoitie, event.x + j - largeurTraitMoitie] = 0
        self.xold = event.x
        self.yold = event.y

    def userInputTest(self, testImage):
        return self.userInputTestTheano(testImage)[0]



if __name__ == '__main__':
    ImageGenerator(10,10)


