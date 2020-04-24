import os
from skimage import io, transform, img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import imageio
from slider_plot import slider_plot
from scipy.special import erfc
import slider_plot
from skimage import color
from analyze_image import fit_all


def conc_func(X, a, b, c, d):
    x, t = X
    return a + (0.5 * c * (1 - erfc((x - b) / (2 * np.sqrt(d*t)))))


def surf_plot(data, path):
    vals = eval(open(path + "/data.txt", "r").read())
    x, y = np.mgrid[0:data.shape[0], 0:data.shape[1]]
    y = y*(.09/(vals["height"][1]-vals["height"][0]))
    ax = plt.axes(projection="3d")
    ax.plot_surface(x, y, data, cmap=plt.cm.viridis)
    plt.xlabel("Time (arb. units)", fontsize=17)
    plt.ylabel("Height (m)", fontsize=17)
    plt.tight_layout()
    plt.savefig(path + "\\plots\\surf_plot.pdf")
    plt.show()


def heatmap_plot(data, color_data, path, time):
    vals = eval(open(path + "/data.txt", "r").read())
    fig, axs = plt.subplots(2, figsize=(18, 10.125))
    x = max(time) - min(time)
    y = np.shape(data)[1]*(.09/(vals["height"][1]-vals["height"][0]))
    aspect = x/(y*3)
    extent = [min(time), max(time), 0, y]
    axs[0].imshow(np.rot90(data), aspect=aspect, extent=extent)
    axs[0].set_title("Gray Scale", fontsize=21)
    axs[0].set_xlabel("Time (seconds)", fontsize=17)
    axs[0].set_ylabel("Height (m)", fontsize=17)
    axs[1].imshow(np.rot90(color_data), aspect=aspect, extent=extent)
    axs[1].set_title("Full Color", fontsize=21)
    axs[1].set_xlabel("Time (seconds)", fontsize=17)
    axs[1].set_ylabel("Height (m)", fontsize=17)
    plt.tight_layout()
    plt.savefig(path + "\\plots\\heatmap_plot.pdf")
    plt.show()


def plot_coeffs(coeffs, time, path):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10.125))

    axes[0, 0].scatter(time, [n[0] for n in coeffs], label=path)
    axes[0, 1].scatter(time, [n[1] for n in coeffs], label=path)
    axes[1, 0].scatter(time, [n[2] for n in coeffs], label=path)
    axes[1, 1].scatter(time, [n[3] for n in coeffs], label=path)

    axes[0, 0].set_title("a", fontsize=19)
    axes[0, 0].set_xlabel("t (seconds)")
    axes[0, 0].set_ylabel("Intensity (arb. units)")

    axes[0, 1].set_title("b", fontsize=19)
    axes[0, 1].set_xlabel("t (seconds)")
    axes[0, 1].set_ylabel("Height (m)")

    axes[1, 0].set_title("c", fontsize=19)
    axes[1, 0].set_xlabel("t (seconds)")
    axes[1, 0].set_ylabel("Intensity (arb. units)")

    axes[1, 1].set_title("d", fontsize=19)
    axes[1, 1].set_xlabel("t (seconds)")
    axes[1, 1].set_ylabel("Arb. units")
    plt.suptitle(
        r"Time dependence of fitting parameters y=$a+0.5\cdot c\cdot$erfc$\left(\frac{x+b}{2\sqrt{d\cdot t}}\right)$",
        fontsize=21)
    plt.savefig(path+"\\plots\\plot.pdf")
    plt.show()


def plot_coeffs_comparison(coeffs, data, color_data, time, path):
    vals = eval(open(path + "/data.txt", "r").read())
    x = max(time) - min(time)
    y = np.shape(data)[1]*(.09/(vals["height"][1]-vals["height"][0]))
    aspect = x / (y * 4)
    extent = [min(time), max(time), 0, y]

    fig, axes = plt.subplots(2, figsize=(18, 10.125), sharex="col")
    axes[0].imshow(np.rot90(color_data), aspect=aspect, extent=extent)
    axes[0].set_title("Full Color", fontsize=19)
    axes[0].set_ylabel("Height (m)", fontsize=17)
    axes[1].scatter(time, [n[0] for n in coeffs])
    axes[1].set_title("a", fontsize=19)
    axes[1].set_xlabel("t (seconds)", fontsize=17)
    axes[1].set_ylabel("Intensity (arb. units)", fontsize=17)
    plt.savefig(path + "\\plots\\a_plot.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, figsize=(18, 10.125), sharex="col")
    axes[0].imshow(np.rot90(color_data), aspect=aspect, extent=extent)
    axes[0].set_title("Full Color", fontsize=19)
    axes[0].set_ylabel("Height (m)", fontsize=17)
    axes[1].scatter(time, [n[1] for n in coeffs])
    axes[1].set_title("b", fontsize=19)
    axes[1].set_xlabel("t (seconds)", fontsize=17)
    axes[1].set_ylabel("Height (m)", fontsize=17)
    plt.savefig(path + "\\plots\\b_plot.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, figsize=(18, 10.125), sharex="col")
    axes[0].imshow(np.rot90(color_data), aspect=aspect, extent=extent)
    axes[0].set_title("Full Color", fontsize=19)
    axes[0].set_ylabel("Height (m)", fontsize=17)
    axes[1].scatter(time, [n[2] for n in coeffs])
    axes[1].set_title("c", fontsize=19)
    axes[1].set_xlabel("t (seconds)", fontsize=17)
    axes[1].set_ylabel("Intensity (arb. units)", fontsize=17)
    plt.savefig(path + "\\plots\\c_plot.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, figsize=(18, 10.125), sharex="col")
    axes[0].imshow(np.rot90(color_data)[:, :], aspect=aspect, extent=extent)
    axes[0].set_title("Full Color", fontsize=19)
    axes[0].set_ylabel("Height (m)", fontsize=17)
    axes[1].scatter(time, [n[3] for n in coeffs])
    axes[1].set_title("d", fontsize=19)
    axes[1].set_xlabel("t (seconds)", fontsize=17)
    axes[1].set_ylabel("Arb. units", fontsize=17)
    plt.savefig(path + "\\plots\\d_plot.pdf")
    plt.close(fig)


def generate_gif(path):
    images = []
    cropped_images = []
    crop, start_time, filetype, angle, height = eval(open(path + "/data.txt", "r").read()).values()
    n = 0
    n_frames = 60
    files = os.listdir(path)
    if n_frames > len(files):
        n_frames = len([n for n in files if n.endswith(filetype)])
    for file in tqdm(files):
        n += 1
        if file.endswith(filetype) and n % int(len(files)/n_frames) == 0:
            im = io.imread(path + "\\" + file, )
            im = transform.rotate(im, float(angle))
            im = im[crop[0]:crop[1], crop[2]:crop[3]]
            cropped_images.append(img_as_ubyte(im))
            images.append(imageio.imread(path+"\\"+file))
    imageio.mimsave(path+"\\plots\\images_gif.gif", images)
    imageio.mimsave(path+"\\plots\\cropped_images_gif.gif", cropped_images)


def plot_fits(data, coeffs, time, path):
    data_vals = eval(open(path + "/data.txt", "r").read())

    def plot_fit(plt, vals):
        n = int(vals[0])
        y = data[n]
        x = np.asarray(range(len(y)))*(.09/(data_vals["height"][1]-data_vals["height"][0]))
        smooth_x = np.linspace(min(x), max(x), len(x)*10)
        plt.plot(smooth_x, conc_func((smooth_x, np.asarray([time[n]]*len(smooth_x))), *coeffs[n]), color="red", label="Fit")
        plt.scatter(x, y, label="Data")
        # plt.title(str(round(time[n], 1))+" Second(s)")
        plt.title(str(coeffs[n]))
    ranges = [{"max": len(time)-1, "min": 0, "step": 1}]
    slider_plot(plot_fit, ranges)


def latex_plot(coeffs, time, path):
    fig, axes = plt.subplots(2, 2, figsize=(8, 4.5))
    axes[0, 0].scatter(time, [n[0] for n in coeffs])
    axes[0, 0].set_title("a", fontsize=19)
    axes[0, 1].scatter(time, [n[1] for n in coeffs])
    axes[0, 1].set_title("b", fontsize=19)
    axes[1, 0].scatter(time, [n[2] for n in coeffs])
    axes[1, 0].set_title("c", fontsize=19)
    axes[1, 1].scatter(time, [n[3] for n in coeffs])
    axes[1, 1].set_title("d", fontsize=19)
    plt.tight_layout()
    plt.savefig(path + "\\plots\\latex_plot.pdf")
    plt.show()


def combined_plot(paths):
    fig, axes = plt.subplots(2, 2, figsize=(18, 10.125))

    for path in paths:
        color_data, time = pickle.load(open(path + "\\plots\\data.p", "rb"))
        data = color.rgb2gray(color_data)
        for n in range(len(data)):
            data[n] = np.clip(data[n], 0, np.average(data[n][-int(len(data[n]) / 10):]))
        coeffs = fit_all(data, time, path)
        a = np.asarray([n[0] for n in coeffs])
        a = a - np.average(a[:10])
        b = np.asarray([n[1] for n in coeffs])
        b = b - np.average(b[:10])
        c = np.asarray([n[2] for n in coeffs])
        c = c - np.average(c[:10])
        d = np.asarray([n[3] for n in coeffs])
        label = path.split("\\")[-1]
        axes[0, 0].scatter(time, a, label=label, s=1)
        axes[0, 1].scatter(time, b, label=label, s=1)
        axes[1, 0].scatter(time, c, label=label, s=1)
        axes[1, 1].scatter(time[30:], d[30:], label=label, s=1)

    axes[0, 0].set_title("a", fontsize=19)
    axes[0, 0].set_xlabel("t (seconds)")
    axes[0, 0].set_ylabel("Intensity (arb. units)")

    axes[0, 1].set_title("b", fontsize=19)
    axes[0, 1].set_xlabel("t (seconds)")
    axes[0, 1].set_ylabel("Height (m)")

    axes[1, 0].set_title("c", fontsize=19)
    axes[1, 0].set_xlabel("t (seconds)")
    axes[1, 0].set_ylabel("Intensity (arb. units)")

    axes[1, 1].set_title("d", fontsize=19)
    axes[1, 1].set_xlabel("t (seconds)")
    axes[1, 1].set_ylabel(r"$\frac{m^2}{s^2}$")
    plt.legend()
    plt.show()
