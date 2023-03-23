from scipy.optimize import curve_fit
import hyperspy.api as hys
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from tqdm import tqdm
plt.rcParams['font.family'] = 'Cambria'

import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap

color_rep = ["black", "red", "green", "blue", "orange", "purple", "yellow", "lime", 
             "cyan", "magenta", "lightgray", "peru", "springgreen", "deepskyblue", 
             "hotpink", "darkgray"]

rgb_rep = {"black":[1,1,1,1], "red":[1,0,0,1], "green":[0,1,0,1], "blue":[0,0,1,1], "orange":[1,0.5,0,1], "purple":[1,0,1,1],
           "yellow":[1,1,0,1], "lime":[0,1,0.5,1], "cyan":[0,1,1,1]}

custom_cmap = mcolors.ListedColormap(color_rep)
bounds = np.arange(-1, len(color_rep))
norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=len(color_rep))
sm = cm.ScalarMappable(cmap=custom_cmap, norm=norm)
sm.set_array([])

cm_rep = ["gray", "Reds", "Greens", "Blues", "Oranges", "Purples"]



class gaussian_deconvolution():
    
    def __init__(self, spectrum_adr, option="spectrum", depth_profile=False, DM_file=True, calib=None):
        
        self.spectrum_adr = spectrum_adr
        
        self.option = option
        self.depth_profile = depth_profile
        
        if option=="spectrum":
            if DM_file:
                num_img = len(spectrum_adr)
                si_data = []
                e_ranges = []
                steps = []
                for i in range(num_img):
                    spectrum = hys.load(spectrum_adr[i], signal_type="EELS")
                    print(spectrum)
                    step = spectrum.axes_manager[0].scale
                    offset = spectrum.axes_manager[0].offset
                    e_size = spectrum.axes_manager[0].size
                    e_range = np.arange(offset, offset+e_size*step, step)
                    si_data.append(spectrum.data)
                    e_ranges.append(e_range)
                    steps.append(step)
                    
            else:
                signals = tifffile.imread(spectrum_adr)
                num_img = len(signals)
                si_data = []
                e_ranges = []
                steps = []
                for i in range(num_img):
                    spectrum = signals[i]
                    step = calib[0]
                    offset = calib[1]
                    e_size = calib[2]
                    e_range = np.arange(offset, offset+e_size*step, step)
                    si_data.append(spectrum)
                    e_ranges.append(e_range)
                    steps.append(step)
            
            self.num_img = num_img
            self.si_data = si_data
            self.e_ranges = e_ranges
            self.steps = steps
                            
        elif option=="SI":
            if depth_profile:
                spectrum = hys.load(spectrum_adr[0], signal_type="EELS")
                print(spectrum)
                spectra = np.mean(spectrum.data, axis=1)
                num_img = len(spectra)
                si_data = []
                e_ranges = []
                steps = []
                for i in range(num_img):
                    step = spectrum.axes_manager[2].scale
                    offset = spectrum.axes_manager[2].offset
                    e_size = spectrum.axes_manager[2].size
                    e_range = np.arange(offset, offset+e_size*step, step)
                    si_data.append(spectra[i])
                    e_ranges.append(e_range)
                    steps.append(step)
                
            else:
                num_img = len(spectrum_adr)
                si_data = []
                e_ranges = []
                steps = []
                for i in range(num_img):
                    spectrum = hys.load(spectrum_adr[i], signal_type="EELS")
                    print(spectrum)
                    step = spectrum.axes_manager[2].scale
                    offset = spectrum.axes_manager[2].offset
                    e_size = spectrum.axes_manager[2].size
                    e_range = np.arange(offset, offset+e_size*step, step)
                    si_data.append(spectrum.data)
                    e_ranges.append(e_range)
                    steps.append(step)
                    
            self.num_img = num_img
            self.si_data = si_data
            self.e_ranges = e_ranges
            self.steps = steps
            
        else:
            print("wrong option!")
            return
        
    def set_fitting(self, start_ev, end_ev, pad, num_gauss, fit_bound):
        self.start_ev = start_ev
        self.end_ev = end_ev
        self.pad = pad
        self.num_gauss = num_gauss
        if len(fit_bound) != num_gauss:
            print("wrong number of gaussian functions")
            return
        
        bound_left = []
        bound_right = []
        for i in [1, 3, 5]:
            for j in range(num_gauss):
                bound_left.append(fit_bound[j][i-1])
                bound_right.append(fit_bound[j][i])
            
        self.bound_left = bound_left
        self.bound_right = bound_right
        
    def fit(self, background="min", result_visual=True, result_print=True):
        
        if self.option=="SI" and self.depth_profile==False:
            pbar = tqdm(total=self.num_img)

            total_maps = []
            area_maps = []
            for k in range(self.num_img):
                e_range = self.e_ranges[k]
                start_ind = find_nearest(e_range, self.start_ev)
                end_ind = find_nearest(e_range, self.end_ev)
                step = self.steps[k]
                
                e_range_int = e_range[start_ind:end_ind]

                params_map = []
                fit_area = []
                SI = self.si_data[k]

                for l in range(SI.shape[0]):
                    for m in range(SI.shape[1]):
                        signal_int = SI[l, m, start_ind:end_ind]
                        signal_int = signal_int / np.max(signal_int)

                        slope = (signal_int[-1] - signal_int[0]) / (e_range_int[-1] - e_range_int[0])
                        intercept = signal_int[0] - slope*e_range_int[0]
                        if background=="min":
                            bg_line = np.full(len(signal_int), np.min(signal_int))
                        elif background=="line":
                            bg_line = slope*e_range_int+intercept
                        else:
                            print("wrong background option")

                        signal_int_bg_removed = signal_int - bg_line
                        signal_int_bg_removed = np.append(np.zeros(self.pad), signal_int_bg_removed)
                        signal_int_bg_removed = np.append(signal_int_bg_removed, np.zeros(self.pad))
                        signal_int_bg_removed = signal_int_bg_removed / np.max(signal_int_bg_removed)
                        e_range_int_bg_removed = np.arange(self.start_ev-self.pad*step, self.end_ev+self.pad*step, step)

                        if background=="min":
                            bg_line_bg_removed = np.full(len(signal_int_bg_removed), np.min(signal_int_bg_removed))
                        elif background=="line":
                            bg_line_bg_removed = slope*e_range_int_bg_removed+intercept / np.max(signal_int_bg_removed)
                        else:
                            print("wrong background option")

                        if self.num_gauss == 2:
                            popt, pcov = curve_fit(two_gauss, e_range_int_bg_removed, signal_int_bg_removed, 
                                               bounds=(self.bound_left, self.bound_right))
                        elif self.num_gauss == 3:
                            popt, pcov = curve_fit(three_gauss, e_range_int_bg_removed, signal_int_bg_removed, 
                                               bounds=(self.bound_left, self.bound_right))            
                        elif self.num_gauss == 4:
                            popt, pcov = curve_fit(four_gauss, e_range_int_bg_removed, signal_int_bg_removed, 
                                               bounds=(self.bound_left, self.bound_right))
                        elif self.num_gauss == 5:
                            popt, pcov = curve_fit(five_gauss, e_range_int_bg_removed, signal_int_bg_removed, 
                                               bounds=(self.bound_left, self.bound_right))                
                        elif self.num_gauss == 6:
                            popt, pcov = curve_fit(six_gauss, e_range_int_bg_removed, signal_int_bg_removed, 
                                               bounds=(self.bound_left, self.bound_right))   
                            
                        params_map.append(popt)
                        
                        fit_point = []
                        fit_result = []
                        total = np.zeros(len(e_range_int_bg_removed))
                        for i in range(self.num_gauss):
                            fitted = gauss(e_range_int_bg_removed, popt[i], popt[i+self.num_gauss], popt[i+self.num_gauss*2])
                            fit_point.append(fitted)
                            total += fitted

                        for i in range(self.num_gauss):
                            fit_result.append(np.trapz(fit_point[i], e_range_int_bg_removed))
                        
                        fit_area.append(fit_result)
                        
                params_map = np.asarray(params_map).reshape(SI.shape[0], SI.shape[1], -1)
                fit_area = np.asarray(fit_area).reshape(SI.shape[0], SI.shape[1], -1)
                total_maps.append(params_map)
                area_maps.append(fit_area)
                pbar.update(1)
            pbar.close()
            
            self.total_maps = total_maps
            self.area_maps = area_maps
            
        else:
            fitting_result = []
            fit_area = []
            
            for k in range(self.num_img):
                e_range = self.e_ranges[k]
                signal = self.si_data[k]
                start_ind = find_nearest(e_range, self.start_ev)
                end_ind = find_nearest(e_range, self.end_ev)
                step = self.steps[k]

                e_range_int = e_range[start_ind:end_ind]
                signal_int = signal[start_ind:end_ind]
                signal_int = signal_int / np.max(signal_int)

                slope = (signal_int[-1] - signal_int[0]) / (e_range_int[-1] - e_range_int[0])
                intercept = signal_int[0] - slope*e_range_int[0]

                if background=="min":
                    bg_line = np.full(len(signal_int), np.min(signal_int))
                elif background=="line":
                    bg_line = slope*e_range_int+intercept
                else:
                    print("wrong background option")

                signal_int_bg_removed = signal_int - bg_line
                signal_int_bg_removed = np.append(np.zeros(self.pad), signal_int_bg_removed)
                signal_int_bg_removed = np.append(signal_int_bg_removed, np.zeros(self.pad))
                signal_int_bg_removed = signal_int_bg_removed / np.max(signal_int_bg_removed)
                e_range_int_bg_removed = np.arange(self.start_ev-self.pad*step, self.end_ev+self.pad*step, step)

                if background=="min":
                    bg_line_bg_removed = np.full(len(signal_int_bg_removed), np.min(signal_int_bg_removed))
                elif background=="line":
                    bg_line_bg_removed = slope*e_range_int_bg_removed+intercept / np.max(signal_int_bg_removed)
                else:
                    print("wrong background option")

                if result_visual:
                    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
                    ax.plot(e_range_int, signal_int, 'k-')
                    ax.plot(e_range_int, bg_line, 'r-')
                    ax.plot(e_range_int_bg_removed, signal_int_bg_removed, 'b-')
                    fig.tight_layout()
                    plt.show()

                if self.num_gauss == 2:
                    popt, pcov = curve_fit(two_gauss, e_range_int_bg_removed, signal_int_bg_removed, 
                                       bounds=(self.bound_left, self.bound_right))
                elif self.num_gauss == 3:
                    popt, pcov = curve_fit(three_gauss, e_range_int_bg_removed, signal_int_bg_removed, 
                                       bounds=(self.bound_left, self.bound_right))            
                elif self.num_gauss == 4:
                    popt, pcov = curve_fit(four_gauss, e_range_int_bg_removed, signal_int_bg_removed, 
                                       bounds=(self.bound_left, self.bound_right))
                elif self.num_gauss == 5:
                    popt, pcov = curve_fit(five_gauss, e_range_int_bg_removed, signal_int_bg_removed, 
                                       bounds=(self.bound_left, self.bound_right))                
                elif self.num_gauss == 6:
                    popt, pcov = curve_fit(six_gauss, e_range_int_bg_removed, signal_int_bg_removed, 
                                       bounds=(self.bound_left, self.bound_right))                

                fitting_result.append(popt)

                fit_point = []
                fit_result = []
                total = np.zeros(len(e_range_int_bg_removed))
                for i in range(self.num_gauss):
                    fitted = gauss(e_range_int_bg_removed, popt[i], popt[i+self.num_gauss], popt[i+self.num_gauss*2])
                    fit_point.append(fitted)
                    total += fitted

                for i in range(self.num_gauss):
                    fit_result.append(np.trapz(fit_point[i], e_range_int_bg_removed))

                fit_area.append(fit_result)

                if result_visual:
                    fig, ax = plt.subplots(1, 2, figsize=(15, 7))
                    for i in range(self.num_gauss):
                        ax[0].plot(e_range_int_bg_removed, signal_int_bg_removed, 'k-')
                        ax[0].fill(e_range_int_bg_removed, fit_point[i], c=color_rep[i+1], alpha=0.3)
                        ax[0].axvline(x = popt[i+self.num_gauss], color=color_rep[i+1], linestyle=":")

                        ax[1].plot(e_range_int_bg_removed, signal_int_bg_removed+bg_line_bg_removed, 'k-')
                        ax[1].fill(e_range_int_bg_removed, fit_point[i]+bg_line_bg_removed, c=color_rep[i+1], alpha=0.3)
                        ax[1].axvline(x = popt[i+self.num_gauss], color=color_rep[i+1], linestyle=":")

                    ax[0].plot(e_range_int_bg_removed, total, 'r*', alpha=0.3)
                    ax[1].plot(e_range_int_bg_removed, total+bg_line_bg_removed, 'r*', alpha=0.3)

                    fig.tight_layout()
                    plt.show()

            self.fitting_result = fitting_result
            self.fit_area = fit_area

            if result_print:
                print("*****************************************************************")
                print("optimized parameters*********************************************")
                print("*****************************************************************")
                for i in range(self.num_img):
                    print(*self.fitting_result[i], sep=" ")
                    
                print("*****************************************************************")    
                print("fit area*********************************************************")
                print("*****************************************************************")
                for i in range(self.num_img):
                    print(*self.fit_area[i], sep=" ")
                    
                    
def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def gauss(x, a, c, sigma):
    return a*np.exp(-(x-c)**2/(2*sigma**2))

def two_gauss(x, a1, a2, c1, c2, sigma1, sigma2):
    return gauss(x, a1, c1, sigma1)+gauss(x, a2, c2, sigma2)

def three_gauss(x, a1, a2, a3, c1, c2, c3, sigma1, sigma2, sigma3):
    return gauss(x, a1, c1, sigma1)+gauss(x, a2, c2, sigma2)+gauss(x, a3, c3, sigma3)

def four_gauss(x, a1, a2, a3, a4, c1, c2, c3, c4, sigma1, sigma2, sigma3, sigma4):
    return gauss(x, a1, c1, sigma1)+gauss(x, a2, c2, sigma2)+gauss(x, a3, c3, sigma3)+gauss(x, a4, c4, sigma4)

def five_gauss(x, a1, a2, a3, a4, a5, c1, c2, c3, c4, c5, sigma1, sigma2, sigma3, sigma4, sigma5):
    return gauss(x, a1, c1, sigma1)+gauss(x, a2, c2, sigma2)+gauss(x, a3, c3, sigma3)+gauss(x, a4, c4, sigma4)+gauss(x, a5, c5, sigma5)

def six_gauss(x, a1, a2, a3, a4, a5, a6, c1, c2, c3, c4, c5, c6, sigma1, sigma2, sigma3, sigma4, sigma5, sigma6):
    return gauss(x, a1, c1, sigma1)+gauss(x, a2, c2, sigma2)+gauss(x, a3, c3, sigma3)+gauss(x, a4, c4, sigma4)+gauss(x, a5, c5, sigma5)+gauss(x, a6, c6, sigma6)

def seven_gauss(x, a1, a2, a3, a4, a5, a6, a7, c1, c2, c3, c4, c5, c6, c7, sigma1, sigma2, sigma3, sigma4, sigma5, sigma6, sigma7):
    return gauss(x, a1, c1, sigma1)+gauss(x, a2, c2, sigma2)+gauss(x, a3, c3, sigma3)+gauss(x, a4, c4, sigma4)+gauss(x, a5, c5, sigma5)+gauss(x, a6, c6, sigma6)+gauss(x, a7, c7, sigma7)