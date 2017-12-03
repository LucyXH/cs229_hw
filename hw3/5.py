from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np


def KMeans(data, n_cluster, iteration):
    m, n = data.shape
    centroids = np.zeros((n_cluster, n))

    # randomply pick up n_cluster centroids
    centroids = data[np.random.choice(m, n_cluster)].astype(float)
    data = data.astype(float)

    for iter in range(iteration):
        c = np.zeros(m)
        # assign a closest centroid for each point
        for i in range(m):
            dist = np.zeros(n_cluster)
            for k in range(n_cluster):
                d = data[i] - centroids[k]
                dist[k] = d.dot(d)
            c[i] = np.argmin(dist)
        # update each centroid by calculating average over all points in
        # this cluster
        for k in range(n_cluster):
            mask = np.vstack(np.array(c == k))
            centroids[k] = 1. * \
                np.sum(data * mask, axis=0) / np.count_nonzero(mask)

        diff = 0.0
        for i in range(m):
            diff += np.linalg.norm(data[i] - centroids[c[i]])

        print("Iter: " + str(iter))
        print(centroids)
        print("Diff: %.3f" % (diff))

    return centroids.astype('uint8')


def compress_image(image, centroids):
    w, h, c = image.shape
    n_cluster = centroids.shape[0]
    img_compress = np.zeros(image.shape)

    for i in range(w):
        for j in range(h):
            dist = np.zeros(n_cluster)
            for k in range(n_cluster):
                d = image[i, j, :].astype(float) - centroids[k].astype(float)
                dist[k] = d.dot(d)
            img_compress[i, j, :] = centroids[np.argmin(dist)]

    return img_compress.astype('uint8')


def main():

    img_small = imread('mandrill-small.tiff')
    img_large = imread('mandrill-large.tiff')
    centroids = KMeans(img_small.reshape(-1, img_small.shape[2]), 16, 30)

    img_small_compress = compress_image(img_small, centroids)
    plt.imsave('compress_small.jpg', img_small_compress)

    img_large_compress = compress_image(img_large, centroids)
    plt.imsave('compress_large.jpg', img_large_compress)


if __name__ == '__main__':
    main()
