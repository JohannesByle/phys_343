from plot_methods import *
from analyze_image import fit_all
import pickle
from scipy.optimize import curve_fit
import imageio
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


def get_intensity_guess_params(x, y, t):

    def find_nearest(array, value):
        array = np.asarray(array)
        value = float(value)
        idx = (np.abs(array - value)).argmin()
        return idx

    y = 1 - y
    c = (np.average(y[-10:]) - np.average(y[:10]))
    a = np.average(y[:10]) + 0.5 * c
    slope = max(np.gradient(y, x))
    b = x[find_nearest(np.gradient(y), slope)]
    d = (c / (2 * np.sqrt(t * np.pi) * slope)) ** 2
    return [a, b, -c, d]


def intensity_fit_all(data, time, path):
    coeffs = []
    vals = eval(open(path + "/data.txt", "r").read())

    for n in range(len(data)):
        y = data[n]
        x = np.asarray(range(len(y))) * (.09 / (vals["height"][1] - vals["height"][0]))
        t = time[n]
        guess = get_intensity_guess_params(x, y, t)
        args, _ = curve_fit(conc_func, (x, np.asarray([t] * len(x))), y, guess)
        coeffs.append(args)

    return coeffs


def import_data(path):
    color_data, time = pickle.load(open(path + "\\plots\\data.p", "rb"))
    data = color.rgb2gray(color_data)
    time = np.asarray(time)

    times = []
    x_arrays = []
    vals = eval(open(path + "/data.txt", "r").read())
    for n in range(len(data)):
        data[n] = np.clip(data[n], 0, np.average(data[n][-int(len(data[n]) / 10):]))
        times.append(np.asarray([time[n]] * len(data[n])))
        x_arrays.append(np.asarray(range(len(data[n]))) * (.09 / (vals["height"][1] - vals["height"][0])))

    coeffs = pickle.load(open(path + "\\plots\\coeffs.p", "rb"))
    # coeffs = fit_all(data, time, path)
    a = np.asarray([n[0] for n in coeffs])
    b = np.asarray([n[1] for n in coeffs])
    c = np.asarray([n[2] for n in coeffs])
    d = np.asarray([n[3] for n in coeffs])

    return data, time, coeffs, a, b, c, d, times, x_arrays


def convert_intensity(y, a):
    return np.exp(-np.asarray(y)*a)


def normalize(y):
    y = np.asarray(y)
    y -= min(y)
    y /= max(y)
    return y


def create_master_gif(path):

    thinning_constant = 20
    y = data[1::thinning_constant]
    x = x_arrays[1::thinning_constant]
    t = times[1::thinning_constant]
    args = coeffs[1::thinning_constant]
    for n in range(len(y)):
        y[n] = conc_func((x[n], t[n]), *args[n])

    def master_gif(fig, ax, args, a):

        n = 0
        y = normalize(data[n])
        x = x_arrays[n]
        ax[1, 0].scatter(x, y, label="Intensity - Data")
        # y = normalize(conc_func((x, times[n]), *coeffs[n]))
        # ax[1, 0].plot(x, y, label="Intensity - Fit", color="orange")
        ax[1, 0].plot(x, normalize(convert_intensity(y, a)), color="red", label="Concentration")
        ax[1, 0].set_title("Relationship between intensity and concentration at " + str(time[n]) + "s")
        ax[1, 0].set_xlabel("Height (m)")
        ax[1, 0].set_ylabel("Intensity or Concentration (arb. units)")

        n = -1
        y = normalize(data[n])
        x = x_arrays[n]
        ax[1, 1].scatter(x, y, label="Intensity - Data")
        # y = normalize(conc_func((x, times[n]), *coeffs[n]))
        # ax[1, 1].plot(x, y, label="Intensity - Fit", color="orange")
        ax[1, 1].plot(x, normalize(convert_intensity(y, a)), color="red", label="Concentration")
        ax[1, 1].set_title("Relationship between intensity and concentration at " + str(time[n]) + "s")
        ax[1, 1].set_xlabel("Height (m)")
        ax[1, 1].set_ylabel("Intensity or Concentration (arb. units)")

        ax[0, 1].plot(y, convert_intensity(y, a))
        ax[0, 1].set_title("Relationship between intensity and concentration")
        ax[0, 1].set_xlabel("Intensity")
        ax[0, 1].set_ylabel("Concentration")

        x = np.asarray([i[0] for i in t][1:])
        y = [i[3] for i in args][1:]
        ax[0, 0].scatter(x, y)
        ax[0, 0].set_title("Value of diffusion coefficient vs time")
        ax[0, 0].set_xlabel("Time (s)")
        ax[0, 0].set_ylabel("Diffusion Coefficient (m$^2/s^2$)")

        plt.suptitle(r"$\exp(-" + str(a) + "x)$")
        plt.legend()
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        return image

    images = []
    a = 0.1
    new_t = [i[0] for i in t]
    while a < 100:
        a += a
        new_y = [normalize(convert_intensity(normalize(n), a)) for n in y]
        args = intensity_fit_all(new_y, new_t, path)
        print(a)
        fig, ax = plt.subplots(2, 2, figsize=(16, 9))
        images.append(master_gif(fig, ax, args, a))
        plt.close(fig)

    imageio.mimsave("plots\\messy_master_gif.gif", images + list(reversed(images)))


def plot_func(x):
    return 5.56/x-np.exp(-4.2*(x-1))+2.8


paths = [r"D:\Coding\phys_343\DarkSyrup_4-7", r"D:\Coding\phys_343\DarkSyrup_4-9", r"\\169.254.83.120\Webcam_Test"]
path = paths[0]

fig, axes = plt.subplots(2, 2, figsize=(16, 9))
for path in paths:
    data, time, coeffs, a, b, c, d, times, x_arrays = import_data(path)
    args, _ = curve_fit(b_func, time, b)
    b_vals = args
    guess = [np.average(a), args[0], args[1], np.average(c), np.average(d)]
    args, stats = curve_fit(conc_func_b, (np.ravel(x_arrays), np.ravel(times)), np.ravel(data), guess)

    # smin = min(time)
    # smax = max(time)
    # srange = smax - smin
    # color_idx = [(n - smin) / srange for n in time]
    #
    # fig = plt.figure(figsize=(16, 9))
    # for n in range(len(data)):
    #     y = conc_func_b((x_arrays[n], times[n]), *args)
    #     if n % int(len(data) / 10) == 0:
    #         plt.plot(x_arrays[n], y, color=cm.rainbow(color_idx[n]), linewidth=1, label=str(round(time[n]/3600, 1))+" h(s)")
    #     else:
    #         plt.plot(x_arrays[n], y, color=cm.rainbow(color_idx[n]), linewidth=1)
    #
    # plt.xlabel("Distance (m)", fontsize=21)
    # plt.ylabel("Intensity (arb. units)", fontsize=21)
    # name = path.split("\\")[-1]
    # if name == "Webcam_Test":
    #     name = "DarkSyrup_4-14"
    # plt.title(name, fontsize=22)
    # plt.legend(fontsize=21)


    # images = []
    #
    # def sample_fit(n):
    #     fig = plt.figure(figsize=(16, 9))
    #     plt.scatter(x_arrays[n], data[n], label="Data")
    #     plt.plot(x_arrays[n], conc_func_b((x_arrays[n], times[n]), *args), color="red", label="Fit")
    #     plt.legend()
    #     plt.xlabel("Height (m)", fontsize=21)
    #     plt.ylabel("Intensity (arb. units)", fontsize=21)
    #     plt.title(str(round(time[n]/3600, 1))+" h(s)")
    #     fig.canvas.draw()
    #     image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    #     image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    #     plt.close(fig)
    #     return image
    #
    # n = 1
    # while n < len(data):
    #     n += int(n/2)+1
    #     print(n)
    #     if n <= len(data):
    #         images.append(sample_fit(n))
    #
    # fig = plt.figure(figsize=(16, 9))
    # r2_data_surface = []
    # r2_data_fit = []
    # for n in tqdm(range(len(data))):
    #     x = x_arrays[n]
    #     y = data[n]
    #     t = times[n]
    #     r2_data_surface.append(r2_score(y, conc_func_b((x, t), *args)))
    #     r2_data_fit.append(r2_score(data[n], conc_func((x_arrays[n], times[n]), *coeffs[n])))
    # plt.plot(time, r2_data_surface)
    # plt.plot(time, r2_data_fit)
    # plt.xlabel("Time (s)", fontsize=21)
    # plt.ylabel("R-squared value", fontsize=21)
    # string = str(round(np.average(r2_data_surface), 3)) + " & "
    # for arg in args:
    #     string += str("{:.2e}".format(arg))+" & "
    # print(string)

    name = path.split("\\")[-1]
    if name == "Webcam_Test":
        name = "DarkSyrup_4-14"
    # plt.title(name, fontsize=22)
    # plt.savefig("plots\\"+name+"rsq_c.png")
    # imageio.mimsave("plots\\"+name+"sample_fit_surf_c.gif", images + list(reversed(images)))

    axes[0, 0].scatter(time, a, label=name)
    axes[0, 1].scatter(time, b, label=name)
    axes[0, 1].plot(time, b_func(time, b_vals[0], b_vals[1]), label=name)
    axes[1, 0].scatter(time, c, label=name)
    axes[1, 1].scatter(time[10:], d[10:], label=name)

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
plt.legend()
plt.savefig("plots\\factors_with_linest.png")
plt.show()

