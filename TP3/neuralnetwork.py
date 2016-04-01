"""
Machine Learning: Handwritten Number Recognition
Dataset: MNIST Data
Inspired from: http://deeplearning.net/tutorial/mlp.html

Eric Morissette (1631103)
Sacha Licatese-Roussel (X)
"""


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
                    dtype = theano.config.floatX), name = 'W', borrow = True)

        # Initialize W as a vector of zeros
        self.b = theano.shared(value = numpy.zeros((n_out,),
                    dtype = theano.config.floatX ), name = 'b', borrow = True)

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyperplane for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of
        # hyperplane-k
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # symbolic description of how to compute prediction as class whose
        # probability is maximal
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)

        # Combine W and B
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # start-snippet-2
        # y.shape[0] is (symbolically) the number of rows in y, i.e.,
        # number of examples (call it n) in the minibatch
        # T.arange(y.shape[0]) is a symbolic vector which will contain
        # [0,1,2,... n-1] T.log(self.p_y_given_x) is a matrix of
        # Log-Probabilities (call it LP) with one row per example and
        # one column per class LP[T.arange(y.shape[0]),y] is a vector
        # v containing [LP[0,y[0]], LP[1,y[1]], LP[2,y[2]], ...,
        # LP[n-1,y[n-1]]] and T.mean(LP[T.arange(y.shape[0]),y]) is
        # the mean (across minibatch examples) of the elements in v,
        # i.e., the mean log-likelihood across the minibatch.
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])


    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """

        # the T.neq operator returns a vector of 0s and 1s, where 1
        # represents a mistake in prediction
        return T.mean(T.neq(self.y_pred, y))

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



def defaultNeuralNetwork():

    # Default Parameters
    learning_rate   = 0.004
    L1_coeff        = 0.00
    L2_coeff        = 0.0001
    n_epochs        = 1000
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

    # Gradient Params
    gradParams = [T.grad(cost, param) for param in classifier.params]

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gradParam)
        for param, gradParam in zip(classifier.params, gradParams)
    ]

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

    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience // 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch

    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.

    epoch = 0
    done_looping = False

    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):

            train_model(minibatch_index)
            # iteration number
            iter = (epoch - 1) * n_train_batches + minibatch_index

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i
                                     in range(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print(
                    'epoch %i, minibatch %i/%i, validation error %f %%' %
                    (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                    )
                )

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:
                    #improve patience if loss improvement is good enough
                    if (
                        this_validation_loss < best_validation_loss *
                        improvement_threshold
                    ):
                        patience = max(patience, iter * patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # test it on the test set
                    test_losses = [test_model(i) for i
                                   in range(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of '
                           'best model %f %%') %
                          (epoch, minibatch_index + 1, n_train_batches,
                           test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    # Training is done
    customPrint('Done', True)

    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))




if __name__ == '__main__':
    defaultNeuralNetwork()