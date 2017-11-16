from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np


def KMeans(data, n_cluster, iteration):
    m, n = data.shape
    centroids = np.zeros((n_cluster, n))

    # randomply pick up n_cluster centroids
    centroids = data[np.random.choice(m, n_cluster)].astype(np.float64)

    for iter in range(iteration):
        c = np.zeros(m)
        # assign a closest centroid for each point
        for i in range(m):
            dist = np.zeros(n_cluster)
            for k in range(n_cluster):
                d = 1. * data[i] - centroids[k]
                dist[k] = d.dot(d)
            c[i] = np.argmin(dist)
        # update each centroid by calculating average over all points in
        # this cluster
        for k in range(n_cluster):
            mask = np.vstack(np.array(c == k))
            centroids[k] = 1. * \
                np.sum(data * mask, axis=0) / np.count_nonzero(mask)

        print("Iter: " + str(iter))
        print(centroids)

    return centroids


def compress_image(image, centroids):
    w, h, c = image.shape
    n_cluster = centroids.shape[0]
    img_compress = np.zeros(image.shape)

    for i in range(w):
        for j in range(h):
            dist = np.zeros(n_cluster)
            for k in range(n_cluster):
                d = image[i, j, :] - centroids[k]
                dist[k] = d.dot(d)
                print(d, dist[k])
            img_compress[i, j, :] = centroids[np.argmin(dist)]

    return img_compress


def main():

    img_small = imread('mandrill-small.tiff')
    img_large = imread('mandrill-large.tiff')
    # centroids = KMeans(img_small.reshape(-1, img_small.shape[2]), 16, 1)
    # centroids = centroids.astype('uint8')
    # print(centroids)
    centroids = np.array([[142, 157, 147], [117, 184, 228],
                          [177, 111, 47], [79, 88, 71],
                          [180, 170, 165], [122, 136, 121],
                          [111, 104, 63], [210, 139, 92],
                          [163, 162, 112], [99, 145, 185],
                          [236, 80, 54], [158, 192, 221],
                          [101, 113, 93], [58, 56, 45],
                          [89, 114, 126], [138, 135, 84]])
    centroids = centroids.astype('uint8')
    # data = img_small.reshape(-1, img_small.shape[2])
    # select = np.random.choice(data.shape[0], 16)
    # print(select)
    # centroids = data[select]
    # print(centroids)

    img_small_compress = compress_image(img_small, centroids)

    plt.imsave('compress_small.jpg', img_small_compress)

    # img_large_compress = compress_image(img_large, centroids)
    # plt.imsave('compress_large.jpg', img_large_compress)


if __name__ == '__main__':
    main()
