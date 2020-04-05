import os
from skimage import io, transform, color
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.special import erfc
from datetime import datetime
from tqdm import tqdm
import imageio


def conc_func(x, a, b, c, d):
    return a + 0.5 * c * (1 - erfc((x - b) / (2 * np.sqrt(d))))


def import_images(path):
    crop, start_time, filetype, angle = eval(open(path + "/data.txt", "r").read()).values()
    start_time = datetime.strptime(start_time, "%Y%m%d%H%M%S")
    data = []
    time = []
    show_image = False
    n = 0
    thinning_constant = 10
    for file in tqdm(os.listdir(path)):
        n += 1
        if file.endswith(filetype) and n % thinning_constant == 0:
            im = io.imread(path + "\\" + file)
            im = transform.rotate(im, float(angle))
            im = im[crop[0]:crop[1], crop[2]:crop[3]]
            if show_image:
                plt.imshow(im)
                plt.show()
            im = np.mean(im, axis=1)
            im = np.flip(im, axis=0)
            data.append(im)
            time.append((datetime.strptime(file[:-len(filetype)], "%Y%m%d%H%M%S") - start_time).total_seconds()/(60*60))

    time, data = zip(*sorted(zip(time, data)))
    return np.asarray(data), time


def surf_plot(data, path):
    x, y = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    ax = plt.axes(projection="3d")
    ax.plot_surface(x, y, data, cmap=plt.cm.viridis)
    plt.xlabel("Time (arb. units)", fontsize=17)
    plt.ylabel("Pixel", fontsize=17)
    plt.tight_layout()
    plt.savefig(path + "\\plots\\surf_plot.pdf")
    plt.show()


def heatmap_plot(data, color_data, path, time):
    fig, axs = plt.subplots(2)
    x = max(time) - min(time)
    y = np.shape(data)[1]
    aspect = x/(y*4)
    extent = [min(time), max(time), 0, y]
    axs[0].imshow(np.rot90(data), aspect=aspect, extent=extent)
    axs[0].set_title("Gray Scale", fontsize=21)
    axs[0].set_xlabel("Time (hours)", fontsize=17)
    axs[0].set_ylabel("Pixel", fontsize=17)
    axs[1].imshow(np.rot90(color_data), aspect=aspect, extent=extent)
    axs[1].set_title("Full Color", fontsize=21)
    axs[1].set_xlabel("Time (hours)", fontsize=17)
    axs[1].set_ylabel("Pixel", fontsize=17)
    plt.tight_layout()
    plt.savefig(path + "\\plots\\heatmap_plot.pdf")
    plt.show()


def fit_all(data):
    coeffs = []
    plot_fit = False
    for y in data:
        x = np.asarray(range(len(y)))
        a = (max(y) - min(y))
        c = a
        b = len(y) / 2
        d = 1
        args, _ = curve_fit(conc_func, x, y, [a, b, c, d])
        coeffs.append(args)
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
    axes[0, 0].set_xlabel("t (hours)")
    axes[0, 0].set_ylabel("Intensity (arb. units)")
    axes[0, 1].scatter(time, [n[1] for n in coeffs])
    axes[0, 1].set_title("b", fontsize=19)
    axes[0, 1].set_xlabel("t (hours)")
    axes[0, 1].set_ylabel("Pixels")
    axes[1, 0].scatter(time, [n[2] for n in coeffs])
    axes[1, 0].set_title("c", fontsize=19)
    axes[1, 0].set_xlabel("t (hours)")
    axes[1, 0].set_ylabel("Intensity (arb. units)")
    axes[1, 1].scatter(time, np.asarray([n[3] for n in coeffs]) / np.asarray(time))
    axes[1, 1].set_title("d", fontsize=19)
    axes[1, 1].set_xlabel("t (hours)")
    axes[1, 1].set_ylabel("Arb. units")
    plt.suptitle(
        r"Time dependence of fitting parameters y=$a+0.5\cdot c\cdot$erfc$\left(\frac{x+b}{2\sqrt{d\cdot t}}\right)$",
        fontsize=21)
    plt.savefig(path+"\\plots\\plot.pdf")
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
    plt.savefig(path + "\\plots\\latex_plot.pdf")
    plt.show()


def generate_gif(path, filetype):
    images = []
    n = 0
    n_frames = 60
    files = os.listdir(path)
    for file in tqdm(files):
        n += 1
        if file.endswith(filetype) and n % int(len(files)/n_frames) == 0:
            images.append(imageio.imread(path+"\\"+file))
    imageio.mimsave(path+"\\plots\\images_gif.gif", images)


path = r"\\169.254.39.157\Shared_Folder\Webcam_Test"
# generate_gif(path, ".jpeg")
color_data, time = import_images(path)
data = color.rgb2gray(color_data)
fit_all(data)
heatmap_plot(data, color_data, path, time)
surf_plot(data, path)
plot_coeffs(fit_all(data), time, path)

