import numpy as np
import matplotlib.pyplot as plt
import pickle

def readData(images_file, labels_file):
    x = np.loadtxt(images_file, delimiter=',')
    y = np.loadtxt(labels_file, delimiter=',')
    return x, y


def softmax(x):
    """
    Compute softmax function for input. 
    Use tricks from previous assignment to avoid overflow
    """
    # YOUR CODE HERE
    (m, n) = x.shape
    s = np.zeros(x.shape)
    for i in range(m):
        for j in range(n):
            x_minum = np.exp(x[i] - x[i][j])
            s[i][j] = 1. / x_minum.sum()
    # END YOUR CODE
    return s


def sigmoid(x):
    """
    Compute the sigmoid function for the input here.
    """
    # YOUR CODE HERE
    s = 1./ (1 + np.exp(-x))
    # END YOUR CODE
    return s


def forward_prop(data, labels, params):
    """
    return hidder layer, output(softmax) layer and loss
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    # YOUR CODE HERE
    z1 = data.dot(W1) + b1
    h = sigmoid(z1)
    z2 = h.dot(W2) + b2
    y = softmax(z2)
    cost = compute_cost(y, labels)
    # END YOUR CODE
    return h, y, cost


def backward_prop(data, labels, params):
    """
    return gradient of parameters
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    # YOUR CODE HERE
    (x, h, y) = data
    m = x.shape[0]
    gradb2 = np.sum(y - labels, axis = 0) / m
    
    gradW2 = np.dot(h.transpose(), y-labels) / m
    
    gradb1 = np.dot(h.transpose(), 1-h).dot(W2)
    gradb1 = np.dot(y-labels, gradb1.transpose())
    gradW1 = np.dot(x.transpose(), gradb1) / m

    gradb1 = np.sum(gradb1, axis = 0) / m
    print(gradW1[0])
    print(gradb1)
    print(gradW2[0])
    print(gradb2)
    # END YOUR CODE

    grad = {}
    grad['W1'] = gradW1
    grad['W2'] = gradW2
    grad['b1'] = gradb1
    grad['b2'] = gradb2

    return grad


def nn_train(trainData, trainLabels, devData, devLabels):
    (m, n) = trainData.shape
    num_hidden = 300
    num_output = 10
    learning_rate = 5
    params = {}

    # YOUR CODE HERE

    # initialize params
    W1 = np.random.standard_normal((n, num_hidden))
    W2 = np.random.standard_normal((num_hidden, num_output))
    b1 = np.zeros(num_hidden)
    b2 = np.zeros(num_output)
    params['W1'] = W1
    params['b1'] = b1
    params['W2'] = W2
    params['b2'] = b2

    loss_train = []
    accuracy_train = []
    loss_val = []
    accuracy_val = []
    num_epochs = 30
    batch_size = 1000
    num_iters = int(m / batch_size)
    for epoch in range(num_epochs):
        loss_epoch_train = 0
        correct_epoch_train = 0
        for iter in range(num_iters):
            input = trainData[iter * batch_size: (iter + 1) * batch_size]
            labels = trainLabels[iter * batch_size: (iter + 1) * batch_size]

            h, output, loss = forward_prop(input, labels, params)
            
            grad = backward_prop((input, h, output), labels, params)
            
            loss_epoch_train += loss * batch_size
            correct_epoch_train += (np.argmax(output, axis=1) == np.argmax(
                labels, axis=1)).sum()

            # update params
            W1 = params['W1']
            b1 = params['b1']
            W2 = params['W2']
            b2 = params['b2']
            gradW1 = grad['W1']
            gradW2 = grad['W2']
            gradb1 = grad['b1']
            gradb2 = grad['b2']
            W1 = W1 - learning_rate * gradW1
            b1 = b1 - learning_rate * gradb1
            W2 = W2 - learning_rate * gradW2
            b2 = b2 - learning_rate * gradb2
            params['W1'] = W1
            params['b1'] = b1
            params['W2'] = W2
            params['b2'] = b2

        loss_train.append(loss_epoch_train / m)
        accuracy_train.append(correct_epoch_train / m)
        print("Epoch %d: training loss: %f, training accuracy: %f" % (epoch+1, loss_train[epoch], accuracy_train[epoch]))

        # test on validation set
        loss_epoch_val, accuracy_epoch_val = nn_test(devData, devLabels, params)
        loss_val.append(loss_epoch_val)
        accuracy_val.append(accuracy_epoch_val)
        print("val loss: %f, val accuracy: %f" % (loss_val[epoch], accuracy_val[epoch]))

    pickle.dump(params, open("params.p", "wb" ))

    # draw
    epochs = range(1, num_epochs + 1)
    plt.figure(111)
    handles = []
    curve1, = plt.plot(epochs, loss_train, label='training loss')
    curve2, = plt.plot(epochs, loss_val, label='validation loss')
    handles.append(curve1)
    handles.append(curve2)
    plt.legend(handles=handles)
    plt.xlabel('number of epochs')
    plt.ylabel('loss')
    plt.title('loss v.s. epochs')
    # plt.show()
    plt.savefig("p1_1.png")

    plt.figure(112)
    handles = []
    curve1, = plt.plot(epochs, accuracy_train, label='training accuracy')
    curve2, = plt.plot(epochs, accuracy_val, label='validation accuracy')
    handles.append(curve1)
    handles.append(curve2)
    plt.legend(handles=handles)
    plt.xlabel('number of epochs')
    plt.ylabel('accuracy')
    plt.title('accuracy v.s. epochs')
    # plt.show()
    plt.savefig('p1_2.png')

    # END YOUR CODE

    return params


def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params)
    accuracy = compute_accuracy(output, labels)
    return cost, accuracy


def compute_accuracy(output, labels):
    accuracy = (np.argmax(output, axis=1) == np.argmax(
        labels, axis=1)).sum() * 1. / labels.shape[0]
    return accuracy


def compute_cost(output, labels):
    y_log = np.log(output)
    cost = 0
    for i in range(output.shape[0]):
        cost += -y_log[i].dot(labels[i]).sum()
    return 1. * cost / output.shape[0]


def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.size, 10))
    one_hot_labels[np.arange(labels.size), labels.astype(int)] = 1
    return one_hot_labels


def main():
    np.random.seed(100)
    # trainData, trainLabels = readData('images_train.csv', 'labels_train.csv')

    # # np.savetxt('images_debug.csv', trainData[:3000], delimiter=',')
    # # np.savetxt('labels_debug.csv', trainLabels[:3000], delimiter=',')
    # # return

    # trainLabels = one_hot_labels(trainLabels)
    # p = np.random.permutation(60000)
    # trainData = trainData[p, :]
    # trainLabels = trainLabels[p, :]

    # devData = trainData[0:10000, :]
    # devLabels = trainLabels[0:10000, :]
    # trainData = trainData[10000:, :]
    # trainLabels = trainLabels[10000:, :]

    trainData, trainLabels = readData('images_debug.csv', 'labels_debug.csv')

    trainLabels = one_hot_labels(trainLabels)
    p = np.random.permutation(3000)
    trainData = trainData[p, :]
    trainLabels = trainLabels[p, :]

    devData = trainData[0:2000, :]
    devLabels = trainLabels[0:2000, :]
    trainData = trainData[2000:, :]
    trainLabels = trainLabels[2000:, :]

    mean = np.mean(trainData)
    std = np.std(trainData)
    trainData = (trainData - mean) / std
    devData = (devData - mean) / std

    testData, testLabels = readData('images_test.csv', 'labels_test.csv')
    testLabels = one_hot_labels(testLabels)
    testData = (testData - mean) / std

    params = nn_train(trainData, trainLabels, devData, devLabels)

    # readyForTesting = False
    # if readyForTesting:
    #     accuracy = nn_test(testData, testLabels, params)
    #     print('Test accuracy: %f' % accuracy)

if __name__ == '__main__':
    main()
