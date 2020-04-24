import pickle
from datetime import datetime
from scipy.optimize import curve_fit
from scipy.signal import savgol_filter
from scipy.special import lambertw
from skimage import color

from plot_methods import *


def import_images(path):
    crop, start_time, filetype, angle, height = eval(open(path + "/data.txt", "r").read()).values()
    start_time = datetime.strptime(start_time, "%Y%m%d%H%M%S")
    data = []
    time = []
    show_image = False
    n = 0
    thinning_constant = 1
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
    c = (np.average(y[-10:]) - np.average(y[:10]))
    a = np.average(y[:10]) + 0.5 * c
    slope = max(np.gradient(y, x))
    b = x[np.where(np.gradient(y, x) == slope)[0][0]]
    d = (c / (2 * np.sqrt(t * np.pi) * slope)) ** 2
    return [a, b, c, d]


paths = [r"\\169.254.83.120\Webcam_Test", r"D:\Coding\phys_343\DarkSyrup_4-7", r"D:\Coding\phys_343\DarkSyrup_4-9"]


exit()

path = paths[3]
# generate_gif(path)
# color_data, time = import_images(path)
# pickle.dump((color_data, time), open(path+"\\plots\\data.p", "wb"))
color_data, time = pickle.load(open(path + "\\plots\\data.p", "rb"))
data = color.rgb2gray(color_data)
for n in range(len(data)):
    data[n] = np.clip(data[n], 0, np.average(data[n][-int(len(data[n]) / 10):]))
coeffs = fit_all(data, time, path)
plot_coeffs_comparison(coeffs, data, color_data, time, path)
plot_fits(data, coeffs, time, path)
# heatmap_plot(data, color_data, path, time)
# surf_plot(data, path)
plot_coeffs(coeffs, time, path)
