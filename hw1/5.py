from __future__ import division
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt


def load_data():
    train = np.genfromtxt('quasar_train.csv',
                          skip_header=True, delimiter=',')
    test = np.genfromtxt('quasar_test.csv',
                         skip_header=True, delimiter=',')
    wavelengths = np.genfromtxt(
        'quasar_train.csv', skip_header=False, delimiter=',')[0]
    return train, test, wavelengths


def add_intercept(X_):
    X = None
    #####################
    X = np.hstack((np.vstack(X_), np.ones((X_.shape[0], 1))))
    ###################
    return X


def smooth_data(raw, wavelengths, tau):
    smooth = None
    ################
    smooth = np.empty(raw.shape)

    for i in range(raw.shape[0]):
        smooth[i] = LWR_smooth(raw[i], wavelengths, tau)
    ################
    return smooth


def LWR_smooth(spectrum, wavelengths, tau):
    smooth_spectrum = None
    ###############
    m = len(wavelengths)
    X = add_intercept(wavelengths)
    smooth_spectrum = np.zeros(m)
    for i in range(m):
        x = wavelengths[i]
        w = np.exp(- (x - wavelengths) * (x - wavelengths)
                   / (2 * tau * tau))
        W = np.diag(w)
        theta = np.linalg.inv(X.transpose().dot(W).dot(X)).dot(
            X.transpose()).dot(W).dot(spectrum)
        smooth_spectrum[i] = theta.dot(X[i])
    ###############
    return smooth_spectrum


def LR_smooth(Y, X_):
    X = add_intercept(X_)
    yhat = np.zeros(Y.shape)
    theta = np.zeros(2)
    #####################
    theta = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(Y)
    yhat = X.dot(theta)
    #####################
    return yhat, theta


def plot_b(X, raw_Y, Ys, desc, filename):
    plt.figure()
    ############
    handles = []
    plt.plot(X, raw_Y, '+')
    for Y, des in zip(Ys, desc):
        curve, = plt.plot(X, Y, label=des)
        handles.append(curve)
    plt.legend(handles=handles)
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    ############
    plt.savefig(filename)


def plot_c(Yhat, Y, X, filename):
    plt.figure()
    ############
    plt.plot(X, Y)
    plt.plot(X[:Yhat.shape[0]], Yhat)
    plt.xlabel('Wavelength')
    plt.ylabel('Flux')
    #############
    plt.savefig(filename)
    return


def split(full, wavelengths):
    left, right = None, None
    ###############
    left_mask = wavelengths < 1200
    right_mask = wavelengths >= 1300
    ln = np.count_nonzero(left_mask)
    rn = np.count_nonzero(right_mask)
    m = full.shape[0]
    left = np.empty((m, ln))
    right = np.empty((m, rn))

    for i in range(m):
        left[i] = full[i][left_mask]
        right[i] = full[i][right_mask]

    ###############
    return left, right


def dist(a, b):
    dist = 0
    ################
    dist = np.sum((a - b) * (a - b))
    ################
    return dist


def func_reg(left_train, right_train, right_test):
    m, n = left_train.shape
    lefthat = np.zeros(n)
    ###########################

    k_indices, h = neighb(right_train, right_test, 3)

    for lamda in range(n):
        w = np.array([ker(dist(right_train[i], right_test) / h)
                      for i in k_indices])
        f = np.array([left_train[i][lamda] for i in k_indices])
        lefthat[lamda] = np.sum(w * f) / np.sum(w)

    ###########################
    return lefthat


def neighb(right_train, right, k):
    dists = [dist(r, right) for r in right_train]
    id_sort = np.argsort(dists)
    k_indices = id_sort[1:k + 1]
    h = dist(right_train[id_sort[-1]], right)
    return k_indices, h


def ker(t):
    if 1 - t > 0:
        return 1 - t
    else:
        return 0


def main():
    raw_train, raw_test, wavelengths = load_data()

    # Part b.i
    lr_est, theta = LR_smooth(raw_train[0], wavelengths)
    print('Part b.i) Theta=[%.4f, %.4f]' % (theta[0], theta[1]))
    plot_b(wavelengths, raw_train[0], [lr_est],
           ['Regression line'], 'ps1q5b1.png')

    # Part b.ii
    lwr_est_5 = LWR_smooth(raw_train[0], wavelengths, 5)
    plot_b(wavelengths, raw_train[0], [lwr_est_5], ['tau = 5'],
           'ps1q5b2.png')

    # Part b.iii
    lwr_est_1 = LWR_smooth(raw_train[0], wavelengths, 1)
    lwr_est_10 = LWR_smooth(raw_train[0], wavelengths, 10)
    lwr_est_100 = LWR_smooth(raw_train[0], wavelengths, 100)
    lwr_est_1000 = LWR_smooth(raw_train[0], wavelengths, 1000)
    plot_b(wavelengths, raw_train[0],
           [lwr_est_1, lwr_est_5, lwr_est_10, lwr_est_100, lwr_est_1000],
           ['tau = 1', 'tau = 5', 'tau = 10', 'tau = 100', 'tau = 1000'],
           'ps1q5b3.png')

    # Part c.i
    smooth_train, smooth_test = [smooth_data(raw, wavelengths, 5)
                                 for raw in [raw_train, raw_test]]

    # Part c.ii
    left_train, right_train = split(smooth_train, wavelengths)
    left_test, right_test = split(smooth_test, wavelengths)

    train_errors = [dist(left, func_reg(left_train, right_train, right))
                    for left, right in zip(left_train, right_train)]
    print('Part c.ii) Training error: %.4f' % np.mean(train_errors))

    # Part c.iii
    test_errors = [dist(left, func_reg(left_train, right_train, right))
                   for left, right in zip(left_test, right_test)]
    print('Part c.iii) Test error: %.4f' % np.mean(test_errors))

    left_1 = func_reg(left_train, right_train, right_test[0])
    plot_c(left_1, smooth_test[0], wavelengths, 'ps1q5c3_1.png')
    left_6 = func_reg(left_train, right_train, right_test[5])
    plot_c(left_6, smooth_test[5], wavelengths, 'ps1q5c3_6.png')
    pass


if __name__ == '__main__':
    main()
