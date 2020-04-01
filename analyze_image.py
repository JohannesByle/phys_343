import os
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from skimage import color
from scipy.optimize import curve_fit
from scipy.special import erfc
from datetime import datetime


def conc_func(x, a, b, c, d):
    return a + 0.5 * c * (1 - erfc((x - b) / (2 * np.sqrt(d))))


def import_images(path, crop, start_time):
    data = []
    time = []
    for file in os.listdir(path):
        if file.endswith(".jpg"):
            im = color.rgb2gray(io.imread(path + "\\" + file))
            im = im[crop[0]:crop[1], crop[2]:crop[3]]
            im = np.flip(np.mean(im, axis=1))
            data.append(im)
            time.append((datetime.strptime(file[:-7], "%Y%m%d%H%M%S") - start_time).total_seconds())

    return np.asarray(data), time


def surf_plot(data, path):
    x, y = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    ax = plt.axes(projection="3d")
    ax.plot_surface(x, y, data, cmap=plt.cm.viridis)
    plt.xlabel("Time (arb. units)", fontsize=17)
    plt.ylabel("Pixel", fontsize=17)
    plt.tight_layout()
    plt.savefig(path + "\\surf_plot.pdf")
    plt.show()


def heatmap_plot(data, path):
    plt.imshow(np.rot90(data), cmap=plt.cm.gray)
    plt.xlabel("Time (arb. units)")
    plt.ylabel("Pixel")
    plt.tight_layout()
    plt.savefig(path + "\\heatmap_plot.pdf")
    plt.show()


def fit_all(data):
    coeffs = []
    for y in data:
        x = np.asarray(range(len(y)))
        a = (max(y) - min(y))
        c = a
        b = len(y) / 2
        d = 1
        args, _ = curve_fit(conc_func, x, y, [a, b, c, d])
        coeffs.append(args)
        plot_fit = False
        if plot_fit:
            plt.scatter(x, y, label="Data", color="red")
            plt.plot(x, conc_func(x, *args), label="Fit")
            plt.xlabel("Height (pixels)", fontsize=17)
            plt.ylabel("Intensity (arb. units)", fontsize=17)
            plt.legend(fontsize=17)
            plt.show()

    return coeffs


def plot_coeffs(coeffs, time, path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10.125))
    axes[0, 0].scatter(time, [n[0] for n in coeffs])
    axes[0, 0].set_title("a", fontsize=19)
    axes[0, 0].set_xlabel("t (seconds)")
    axes[0, 0].set_ylabel("Intensity (arb. units)")
    axes[0, 1].scatter(time, [n[1] for n in coeffs])
    axes[0, 1].set_title("b", fontsize=19)
    axes[0, 1].set_xlabel("t (seconds)")
    axes[0, 1].set_ylabel("Pixels")
    axes[1, 0].scatter(time, [n[2] for n in coeffs])
    axes[1, 0].set_title("c", fontsize=19)
    axes[1, 0].set_xlabel("t (seconds)")
    axes[1, 0].set_ylabel("Intensity (arb. units)")
    axes[1, 1].scatter(time, np.asarray([n[3] for n in coeffs]) / np.asarray(time))
    axes[1, 1].set_title("d", fontsize=19)
    axes[1, 1].set_xlabel("t (seconds)")
    axes[1, 1].set_ylabel("Arb. units")
    plt.suptitle(
        r"Time dependence of fitting parameters y=$a+0.5\cdot c$erfc$\left(\frac{x+b}{2\sqrt{d\cdot t}}\right)$",
        fontsize=21)
    plt.savefig(path+"\\plot.pdf")
    plt.show()


def latex_plot(coeffs, time, path):
    fig, axes = plt.subplots(2, 2, figsize=(8, 4.5))
    axes[0, 0].scatter(time, [n[0] for n in coeffs])
    axes[0, 0].set_title("a", fontsize=19)
    axes[0, 1].scatter(time, [n[1] for n in coeffs])
    axes[0, 1].set_title("b", fontsize=19)
    axes[1, 0].scatter(time, [n[2] for n in coeffs])
    axes[1, 0].set_title("c", fontsize=19)
    axes[1, 1].scatter(time, np.asarray([n[3] for n in coeffs]) / np.asarray(time))
    axes[1, 1].set_title("d", fontsize=19)
    plt.tight_layout()
    plt.savefig(path + "\\latex_plot.pdf")
    plt.show()


path = r"D:/Coding/phys_343/DarkSyrup_4-1"
crop, start_time = eval(open(path + "/data.txt", "r").read()).values()
start_time = datetime.strptime(start_time, "%Y%m%d%H%M%S")
data, time = import_images(path, crop, start_time)
fit_all(data)
# heatmap_plot(data, path)
plot_coeffs(fit_all(data), time, path)
# latex_plot(fit_all(data), time, path)
