import numpy as np
import matplotlib.pyplot as plt
import src.resnet_main as res
import time
from scipy.stats import linregress
from pylab import *

if __name__ == "__main__":
    # # PARAMS:
    ALBUM_CREATED = 11
    JUMP = 5
    PASS_INPUT = "C:\MyWork\AutomaticPhotoAlbum\etgar9_images\Test_input"
    PASS_OUTPUT = "C:\MyWork\AutomaticPhotoAlbum\etgar9_images\Test_output"
    album_name = "Etgar Graduation Ceremony"

    clusters_numbers = [x * JUMP for x in range(1, ALBUM_CREATED)]
    clusters_numbers.insert(0, 1)

    elapsed_time = []

    for num in clusters_numbers:
        print("Creating album number " + (num // JUMP + 1).__str__() + "/" + (ALBUM_CREATED).__str__())
        start = time.time()
        res.create_album(album_name, PASS_INPUT, PASS_OUTPUT, num)
        end = time.time()
        elapsed_time.append(end - start)

    # calculate a linear formula
    linregress(clusters_numbers, elapsed_time)

    plt.plot(clusters_numbers, elapsed_time)

    x_arr = np.asarray(clusters_numbers)
    y_arr = np.asarray(elapsed_time)

    # creating linear model
    slope, intercept, r_value, p_value, std_err = linregress(x_arr, y_arr)
    # creating scatterPlot
    fig = plt.figure()
    gca().set_position((.1, .3, .8, .6))  # to make a bit of room for extra text
    plt.plot(x_arr, y_arr, 'o', label=' album (data point)')
    plt.title('Time to create album as a function of the number of clusters in the album')
    plt.plot(x_arr, intercept + slope * x_arr, 'r', label='linear regression')
    plt.ylabel('Time to create album (in seconds)')
    plt.xlabel('Number of cluster created')
    plt.legend()
    fig.text(.2, .12, "Figure discription : \n The linear model is Time = "
             + slope.__str__()
             + " * num_of_clusters + " + intercept.__str__()
             + "\n R^2 = " + r_value.__str__() + "\n and the Standard deviation = " + std_err.__str__())

    plt.show()
