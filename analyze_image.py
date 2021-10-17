import pickle
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy.signal import savgol_filter
from scipy.special import lambertw
from skimage import color
import numpy as np
from plot_methods import conc_func
from skimage import io
from skimage import transform
import matplotlib.pyplot as plt
from tqdm import tqdm
from plot_methods import *
from sklearn.metrics import r2_score
from emcee_function import run_emcee


def import_images(path):
    crop, start_time, filetype, angle, height = eval(open(path + "/data.txt", "r").read()).values()
    start_time = datetime.strptime(start_time, "%Y%m%d%H%M%S")
    data = []
    time = []
    show_image = False
    n = 0
    thinning_constant = 1
    for file in tqdm(sorted(os.listdir(path))):
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
            time.append((datetime.strptime(file[:-len(filetype)], "%Y%m%d%H%M%S") - start_time).total_seconds())

    time, data = zip(*sorted(zip(time, data)))
    return np.asarray(data), time


def fit_all(data, time, path):
    coeffs = []
    vals = eval(open(path + "/data.txt", "r").read())

    for n in range(len(data)):
        y = data[n]
        x = np.asarray(range(len(y))) * (.09 / (vals["height"][1] - vals["height"][0]))
        t = time[n]
        guess = get_guess_params(x, y, t)
        args, _ = curve_fit(conc_func, (x, np.asarray([t] * len(x))), y, guess)
        coeffs.append(args)

    return coeffs


def get_guess_params(x, y, t):

    def find_nearest(array, value):
        array = np.asarray(array)
        value = float(value)
        idx = (np.abs(array - value)).argmin()
        return idx

    c = (np.average(y[-10:]) - np.average(y[:10]))
    a = np.average(y[:10]) + 0.5 * c
    slope = max(np.gradient(y, x))
    b = x[find_nearest(y, slope)]
    d = (c / (2 * np.sqrt(t * np.pi) * slope)) ** 2
    return [a, b, c, d]


# paths = [r"D:\Coding\phys_343\DarkSyrup_4-7", r"\\169.254.83.120\Webcam_Test", r"D:\Coding\phys_343\DarkSyrup_4-9"]
#
# coeffs = pickle.load(open("coeffs.p", "rb"))
#
#
# # def plot_coeff(plt, val):
# #     n = int(val[0])
# #     d = np.asarray([i[3] for i in coeffs[n]])
# #     plt.plot(range(len(d)), d)
# #
# #
# # ranges = [{"max": len(coeffs) - 1, "min": 0, "step": 1}]
# # slider_plot(plot_coeff, ranges)
# # exit()
# for path in paths:
#     color_data, time = pickle.load(open(path + "\\plots\\data.p", "rb"))
#     time = np.asarray(time)
#     # coeffs = pickle.load(open(path + "\\plots\\coeffs.p", "rb"))
#
#     times = []
#     x_arrays = []
#     data = color.rgb2gray(color_data)
#     vals = eval(open(path + "/data.txt", "r").read())
#     for n in range(len(data)):
#         data[n] = np.clip(data[n], 0, np.average(data[n][-int(len(data[n]) / 10):]))
#
#         times.append(np.asarray([time[n]] * len(data[n])))
#         x_arrays.append(np.asarray(range(len(data[n]))) * (.09 / (vals["height"][1] - vals["height"][0])))
#     coeffs_list = []
#
#     a = np.asarray([n[0] for n in coeffs])
#     b = np.asarray([n[1] for n in coeffs])
#     c = np.asarray([n[2] for n in coeffs])
#     d = np.asarray([n[3] for n in coeffs])
#     plot_fits(data, coeffs, time, path)
#     plot_coeffs(coeffs, time, path)
#
#     exit()
#     guess = [np.average(a), np.average(b), np.average(c), np.average(d)]
#     args, stats = curve_fit(conc_func, (np.ravel(x_arrays), np.ravel(times)), np.ravel(data))
#     # args = [3.69884670e-01, 2.50564117e-02, 4.63361831e-01, 1.85383469e-10]
#     r2_data_surface = []
#     r2_data_guess = []
#     r2_data_fit = []
#     for n in tqdm(range(len(data))):
#         x = x_arrays[n]
#         y = data[n]
#         t = times[n]
#         r2_data_surface.append(r2_score(y, conc_func((x, t), *args)))
#         guess = get_guess_params(x, y, t[0])
#         r2_data_guess.append(r2_score(data[n], conc_func((x_arrays[n], times[n]), *guess)))
#         fit, _ = curve_fit(conc_func, (x, t), y, guess)
#         r2_data_fit.append(r2_score(data[n], conc_func((x_arrays[n], times[n]), *fit)))
#
#     plt.scatter(time, r2_data_fit, label="fit")
#     plt.scatter(time, r2_data_guess, label="guess")
#     plt.scatter(time, r2_data_surface, label="surf")
#     plt.legend()
#     plt.show()
#
# exit()
#
# # # ax = plt.axes(projection="3d")
# # # ax.plot_surface(x_arrays, times, np.reshape(conc_func((np.ravel(x_arrays), np.ravel(times)), *args), np.shape(x_arrays)), cmap=plt.cm.viridis)
# # plt.show()
# #
# # exit()
# # data_sets = []
# # for path in paths:
# #     color_data, time = pickle.load(open(path + "\\plots\\data.p", "rb"))
# #     data = color.rgb2gray(color_data)
# #     for n in range(len(data)):
# #         data[n] = np.clip(data[n], 0, np.average(data[n][-int(len(data[n]) / 10):]))
# #     coeffs = fit_all(data, time, path)
# #     data_sets.append((coeffs, time, path))
# # combined_plot(data_sets)
#
# paths = [r"\\169.254.83.120\Webcam_Test", r"D:\Coding\phys_343\DarkSyrup_4-7", r"D:\Coding\phys_343\DarkSyrup_4-9"]
# for path in paths:
#     generate_gif(path)
#     color_data, time = import_images(path)
#     pickle.dump((color_data, time), open(path + "\\plots\\data.p", "wb"))
#     color_data, time = pickle.load(open(path + "\\plots\\data.p", "rb"))
#     data = color.rgb2gray(color_data)
#     for n in range(len(data)):
#         data[n] = np.clip(data[n], 0, np.average(data[n][-int(len(data[n]) / 10):]))
#     coeffs = fit_all(data, time, path)
#     pickle.dump(coeffs, open(path + "\\plots\\coeffs.p", "wb"))
# # plot_coeffs_comparison(coeffs, data, color_data, time, path)
# # plot_fits(data, coeffs, time, path)
# # # heatmap_plot(data, color_data, path, time)
# # # surf_plot(data, path)
# # plot_coeffs(coeffs, time, path)
