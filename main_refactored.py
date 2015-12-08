import math
import numpy as np
import pylab
import scipy.fftpack
import scipy.interpolate
import scipy.io
import h5py

__author__ = 'alfredoc'

###############################################
######## AUXILIARY HANDMADE FUNCTIONS #########
###############################################


def detect_peak(y_axis, x_axis=None, lookahead=300, delta=0):
    """
    Converted from/based on a MATLAB script at:
    http://billauer.co.il/peakdet.html

    function for detecting local maximas and minmias in a signal.
    Discovers peaks by searching for values which are surrounded by lower
    or larger values for maximas and minimas respectively

    keyword arguments:
    y_axis -- A list containg the signal over which to find peaks
    x_axis -- (optional) A x-axis whose values correspond to the y_axis list
        and is used in the return to specify the postion of the peaks. If
        omitted an index of the y_axis is used. (default: None)
    lookahead -- (optional) distance to look ahead from a peak candidate to
        determine if it is the actual peak (default: 200)
        '(sample / period) / f' where '4 >= f >= 1.25' might be a good value
    delta -- (optional) this specifies a minimum difference between a peak and
        the following points, before a peak may be considered a peak. Useful
        to hinder the function from picking up false peaks towards to end of
        the signal. To work well delta should be set to delta >= RMSnoise * 5.
        (default: 0)
            delta function causes a 20% decrease in speed, when omitted
            Correctly used it can double the speed of the function

    return -- two lists [max_peaks, min_peaks] containing the positive and
        negative peaks respectively. Each cell of the lists contains a tupple
        of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*tab)
    """
    max_peaks = []
    min_peaks = []
    dump = []   #Used to pop the first hit which almost always is false

    # check input data
    x_axis, y_axis = datacheck_peakdetect(x_axis, y_axis)
    # store data length for later use
    length = len(y_axis)


    #perform some checks
    if lookahead < 1:
        raise ValueError("Lookahead must be '1' or above in value")
    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")

    #maxima and minima candidates are temporarily stored in
    #mx and mn respectively
    mn, mx = np.Inf, -np.Inf

    #Only detect peak if there is 'lookahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-lookahead],
                                        y_axis[:-lookahead])):
        if y > mx:
            mx = y
            mxpos = x
        if y < mn:
            mn = y
            mnpos = x

        ####look for max####
        if y < mx-delta and mx != np.Inf:
            #Maxima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].max() < mx:
                max_peaks.append([mxpos, mx])
                dump.append(True)
                #set algorithm to only find minima now
                mx = np.Inf
                mn = np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
                continue
            #else:  #slows shit down this does
            #    mx = ahead
            #    mxpos = x_axis[np.where(y_axis[index:index+lookahead]==mx)]

        ####look for min####
        if y > mn+delta and mn != -np.Inf:
            #Minima peak candidate found
            #look ahead in signal to ensure that this is a peak and not jitter
            if y_axis[index:index+lookahead].min() > mn:
                min_peaks.append([mnpos, mn])
                dump.append(False)
                #set algorithm to only find maxima now
                mn = -np.Inf
                mx = -np.Inf
                if index+lookahead >= length:
                    #end is within lookahead no more peaks can be found
                    break
            #else:  #slows shit down this does
            #    mn = ahead
            #    mnpos = x_axis[np.where(y_axis[index:index+lookahead]==mn)]


    #Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        #no peaks were found, should the function return empty lists?
        pass

    return [max_peaks, min_peaks]


def datacheck_peakdetect(x_axis, y_axis):
    if x_axis is None and y_axis is not None:
        x_axis = range(len(y_axis))

    if y_axis is None:
        raise (ValueError, 'Input vector y_axis cannot be "None"')
    elif len(y_axis) != len(x_axis):
        raise (ValueError, 'Input vectors y_axis and x_axis must have same length')

    # Convert to numpy array if the input is a list
    x_axis, y_axis  = np.array(x_axis), np.array(y_axis)
    return x_axis, y_axis


def get_spectrum(original_signal):
    # Function getting the FFT array and its associated frequencies
    """
    Receives a real time signal y(t). Returns the transformed signal Y(f) and the frequency axis f.
    """
    time_vector, time_domain_signal = original_signal[0], original_signal[1]

    sampling_period = time_vector[1] - time_vector[0]
    number_of_samples = time_domain_signal.size
    n_fft = int(2 ** math.ceil(math.log(number_of_samples, 2)))  # Find the next power of 2
    frequency_domain_signal = scipy.fftpack.fft(time_domain_signal, n_fft) / (number_of_samples / 2)
    frequency_vector = scipy.fftpack.fftfreq(n_fft, sampling_period)

    fft_signal = [frequency_vector, frequency_domain_signal]

    return fft_signal


def choose_time_interval(long_signal, long_pip, start_time, end_time):
    # Shorten the signal to the specified time interval
    start_index = np.where(np.diff(np.sign(long_signal[0] - start_time)))[0]
    end_index = np.where(np.diff(np.sign(long_signal[0] - end_time)))[0]
    range_indices = np.arange(start_index, end_index)
    signal = long_signal[1][np.arange(start_index, end_index)]
    time = long_signal[0][range_indices] - long_signal[0][start_index]
    pip = long_pip[1][range_indices]
    return signal, pip, time


def get_averaged_cycle(signal, pip, time, plot_all_cycles=False):
    """
    Receives a vibration amplitude vector, the corresponding time vector and a revolution indicator pip.
    Returns an average of the combustion cycles in the selected interval normalized to degrees.
    """

    # Get arrays with the indices for every second 'pip' signal
    pip_peaks_list = detect_peak(pip, lookahead=1e3, delta=2e0)[0]
    pip_peaks = {'x': np.array([], dtype=int), 'y': np.array([])}
    for peak in pip_peaks_list:
        pip_peaks['x'] = np.append(pip_peaks['x'], peak[0])
        pip_peaks['y'] = np.append(pip_peaks['y'], peak[1])
    pip_peak_pairs = pip_peaks['x'][0::2]
    # pip_peak_pairs = pip_peaks['x'][1::2]


    # Normalize each cycle to 720 degrees, interpolate to a common resolution and compute an average cycle
    number_of_samples = 8192 #  3600
    average_pressure = np.zeros(number_of_samples)
    number_of_cycles = len(pip_peak_pairs) - 1
    for cycle_ in range(number_of_cycles):
        # noinspection PyTypeChecker
        cycle_indices = np.arange(pip_peak_pairs[cycle_], pip_peak_pairs[cycle_ + 1], dtype=int)
        cycle_time = time[cycle_indices] - time[cycle_indices[0]]  #  Normalizing so t_0 = 0 for this cycle
        cycle_time_deg = cycle_time * (720.0 / cycle_time[-1])
        cycle_pressure = scipy.interpolate.interp1d(cycle_time_deg, signal[cycle_indices])
        cycle_angle = np.linspace(0, 720, num=number_of_samples, endpoint=False)
        single_cycle = cycle_pressure(cycle_angle)
        average_pressure += single_cycle
    average_pressure /= number_of_cycles
    average_cycle = [cycle_angle, average_pressure]

    if plot_all_cycles:
        pylab.figure
        for cycle_ in range(number_of_cycles):
            pylab.plot(cycle_angle, single_cycle)
        pylab.suptitle("Combustion cycles before averaging", fontsize=20)
        pylab.xlabel('Angle ref. pip signal [deg]')
        pylab.ylabel('{0}{1}'.format('Pressure', ' [MPa]'))
        pylab.xlim([0, 720])
    else:
        pass

    return average_cycle


def apply_time_offset(original_pair, offset_deg):
    time = original_pair[0]
    original_signal = original_pair[1]
    offset_index = np.where(np.diff(np.sign(time - offset_deg)))[0]
    shifted_signal = original_signal[range(len(original_signal))-offset_index]
    shifted_pair = [time, shifted_signal]
    return shifted_pair


def get_tangential_pressure(cylinder_pressure_cycle, conrod_ratio):
    # This function makes use of kinematic considerations for converting raw pressure data into tangential pressure
    # Tangential Pressure = Cylinder Pressure * SIN(Crank Angle) * { 1 + lambda * COS(Crank Angle) / SQRT[1 - {lambda * COS(Crank Angle)}**2] }
    # where  = Crank Radius / Connecting Rod Length

    crank_angle = np.radians(cylinder_pressure_cycle[0])
    cylinder_pressure = cylinder_pressure_cycle[1]

    modulation = np.sin(crank_angle) * (
        1 + conrod_ratio * np.cos(crank_angle) / np.sqrt(1 - (conrod_ratio * np.cos(crank_angle)) ** 2))
    tangential_pressure = cylinder_pressure * modulation

    tangential_cycle = [np.degrees(crank_angle), tangential_pressure]
    tangential_modulation = [np.degrees(crank_angle), modulation]
    return tangential_cycle, tangential_modulation


def get_instant_torque(tangential_pressure, cyl_diameter, crank_radius, number_of_cylinders=16):

    crank_angle = tangential_pressure[0]
    torque = tangential_pressure[1] * math.pi/4 * (cyl_diameter**2) * crank_radius * number_of_cylinders

    instant_torque = [crank_angle, torque]
    return instant_torque


def get_excitation_orders(signal, rpm, max_order, plot_excitation_orders=False):

    number_of_cycles = 16
    number_of_samples = len(signal[0]) * number_of_cycles
    crank_angle, rotating_frequency = signal[0], rpm/60

    time_one_cycle = crank_angle/(360*rotating_frequency)
    sampling_period = time_one_cycle[1] - time_one_cycle[0]
    stop_time = (time_one_cycle[-1] + sampling_period) * number_of_cycles
    extended_time = np.linspace(0, stop_time, num=number_of_samples, endpoint=False)
    extended_signal = np.tile(signal[1], number_of_cycles)

    # 1- Find index for frequency equal to 1.1*rotating_frequency*last_order and discard values above that
    excitation_fft = get_spectrum([extended_time, extended_signal])
    cutoff_frequency = 1.1 * rotating_frequency * max_order
    end_index = np.where(np.diff(np.sign(excitation_fft[0] - cutoff_frequency)))[0][0]
    short_excitation_fft = [excitation_fft[0][0:end_index], excitation_fft[1][0:end_index]]

    # 2- Find peaks and save the relevant information associated to them
    peak_index_list = np.nonzero(short_excitation_fft[1])[0]
    excitation_peaks = {'order': np.array([]), 'freq': np.array([]), 'abs': np.array([]), 'real': np.array([]), 'imag': np.array([])}
    for peak in peak_index_list:
        excitation_peaks['order'] = np.append(excitation_peaks['order'], 0.5 * (len(excitation_peaks['order'])))
        excitation_peaks['freq'] = np.append(excitation_peaks['freq'], short_excitation_fft[0][peak])
        excitation_peaks['abs'] = np.append(excitation_peaks['abs'], abs(short_excitation_fft[1][peak]))
        excitation_peaks['real'] = np.append(excitation_peaks['real'], (short_excitation_fft[1][peak]).real)
        excitation_peaks['imag'] = np.append(excitation_peaks['imag'], (short_excitation_fft[1][peak]).imag)

    # freq_resolution = short_excitation_fft[0][1] - short_excitation_fft[0][0]
    # print("freq_resolution = %r" % freq_resolution)
    # print("1 / freq_resolution = %r" % (1.0 / freq_resolution))
    # print("n_samples = %r " % len(excitation_fft[0]))
    # print("freq_resolution * n_samples = %r" % (freq_resolution * len(excitation_fft[0])))
    # print("1 / freq_resolution * n_samples = %r\n\n\n" % (1.0 / freq_resolution * len(excitation_fft[0])))
    # print("\n\n\nshort_excitation_fft[0] = %r" % short_excitation_fft[0])
    # print("\n\n\nabs(short_excitation_fft[1]) = %r" % abs(short_excitation_fft[1]))
    # print("peak_index_list = %r" % peak_index_list)

    if plot_excitation_orders:
        pylab.figure(3)
        pylab.plot(short_excitation_fft[0], abs(short_excitation_fft[1]))
        pylab.plot(excitation_peaks['freq'], excitation_peaks['abs'], '.g')  # Green dots on excitation peaks

    return excitation_peaks


###############################################
################# IMPORT DATA #################
###############################################
file_name = 'data_100pct.h5'
data = {}
with h5py.File(file_name, 'r') as h5f:
    for field in list(h5f):
        data[field] = h5f[field].value

# Import the relevant variables to Python
signal_yt = [data['time'], data['vibration_amplitude']]
number_of_points = data['number_of_points']
PIP = [data['time'], data['pip']]
sample_rate = data['sample_rate']


###############################################
################ PROCESS DATA #################
###############################################

start_time, end_time = 50, 55

# 1- NORMALIZE TO 720 DEGREES, INTERPOLATE TO A COMMON RESOLUTION AND GET AVERAGE CYCLE
signal, pip, time = choose_time_interval(signal_yt, PIP, start_time, end_time)
one_pressure_cycle = get_averaged_cycle(signal, pip, time, plot_all_cycles=False)

# 2- SHIFT SIGNAL REFERENCE TO tdc=0. ADD AN OFFSET TO GET ZERO PRESSURE AT 540deg
angle_tdc = 58.8  #  Deg
shifted_pressure_cycle = apply_time_offset(one_pressure_cycle, angle_tdc)
# pressure_offset = 2.360 + 0.935  #  = 3.295 [MPa]
pressure_offset = 0.270 + 0.935 #  = 1.205 [MPa]
shifted_pressure_cycle[1] += pressure_offset

# 3- CONVERTING TO 'TANGENTIAL PRESSURE'
crank_throw, connecting_rod_length = 0.1075, 0.3925  #  [m]
conrod_ratio = crank_throw / connecting_rod_length
tangential_pressure, tangential_modulation = get_tangential_pressure(shifted_pressure_cycle, conrod_ratio)

# 4- CONVERTING TO TORQUE
cylinder_diameter = 0.170 #  [m]
instant_torque = get_instant_torque(tangential_pressure, cylinder_diameter, crank_throw, number_of_cylinders = 16)

# 5- FFT FOR AN INFINITE NUMBER OF CYCLES
tangential_pressure_orders = get_excitation_orders(tangential_pressure, rpm=1500, max_order=9, plot_excitation_orders=False)
torque_orders = get_excitation_orders(instant_torque, rpm=1500, max_order=9, plot_excitation_orders=False)

nauticus_pressure_orders = [25*np.array([0.5, 1.00, 1.5000, 2.000, 2.5000, 3.5000, 4.0000, 4.5000, 5.0000, 5.5000, 6.0000, 6.5000, 7.0000, 7.5000, 8.0000, 8.5000]),
                            np.array([0.795, 1.345, 1.0451, 0.977, 0.8191, 0.5651, 0.4221, 0.3561, 0.2861, 0.2221, 0.1731, 0.1313, 0.0996, 0.0769, 0.0588, 0.0434])]


###############################################
################# PRINT OUTPUT ################
###############################################
test_figure = pylab.figure(num=1)

pylab.subplot(3, 1, 1)
# pylab.plot(one_pressure_cycle[0], one_pressure_cycle[1], label='Average cycle')
pylab.plot(shifted_pressure_cycle[0], shifted_pressure_cycle[1], 'b', label='Average cycle (tdc=0)')
pylab.plot(tangential_modulation[0], tangential_modulation[1]*10, '--g', label='Tangential modulation')
pylab.plot(tangential_pressure[0], tangential_pressure[1], 'r', label='Tangential pressure [MPa]')
pylab.legend(('Average cycle (tdc=0)', 'Tangential modulation (x10)', 'Tangential pressure [MPa]'), loc='best')
pylab.title("Cylinder pressure {0}%".format(file_name[5:8]))
pylab.xticks(np.arange(0, 720, 90))
pylab.xlabel('Angle ref. FTDC [deg]')
pylab.ylabel('{0}{1}'.format('Pressure', ' [MPa]'))
pylab.xlim([0, 720])

pylab.subplot(3, 1, 2)
pylab.plot(tangential_pressure[0], tangential_pressure[1], 'r', label='Tangential pressure [MPa]')
# pylab.legend(('Tangential pressure [MPa]'), loc='best')
pylab.title("Tangential pressure for one cylinder")
pylab.xticks(np.arange(0, 720, 90))
pylab.xlabel('Angle ref. FTDC [deg]')
pylab.ylabel('{0}{1}'.format('Tangential pressure', ' [MPa]'))
pylab.xlim([0, 720])

pylab.subplot(3, 1, 3)
pylab.plot(tangential_pressure_orders['freq'], tangential_pressure_orders['abs'], 'sr', label='Measurements [MPa]')
pylab.plot(nauticus_pressure_orders[0], nauticus_pressure_orders[1], 'sb', label='Nauticus Machinery [MPa]')
pylab.legend(('Measurements [MPa]', 'Nauticus Machinery [MPa]'), loc='best')
for index_ in range(len(tangential_pressure_orders['freq'])):
    pylab.text(tangential_pressure_orders['freq'][index_], tangential_pressure_orders['abs'][index_]+0.05,
               str(tangential_pressure_orders['order'][index_]), fontsize=12, color='r')
pylab.title("Harmonic table")
pylab.xlabel('Signal frequency [Hz]')
pylab.ylabel('{0}{1}'.format('Tangential pressure harmonics', ' [MPa]'))

pylab.show(test_figure)


###############################################
################# EXPORT DATA #################
###############################################
the_file_name = 'my_file1.txt'
the_file = open(the_file_name, 'w')

the_file.write("{0} {1} {2} {3}\n".format('Angle[deg]', 'Tangential_modulation', 'Pressure_cycle_[MPa]', 'Tangential_pressure[MPa]'))
for item in range(len(tangential_modulation[0])):
    the_file.write("{0} {1} {2} {3}\n".format(tangential_modulation[0][item], tangential_modulation[1][item], shifted_pressure_cycle[1][item], tangential_pressure[1][item]))
the_file.close()
print("Data from exported to file {0}".format(the_file_name))


the_file_name = 'my_file2.txt'
the_file = open(the_file_name, 'w')

the_file.write("{0} {1} {2} {3}\n".format('Frequency[Hz]', 'Measured_harmonics_[MPa]_abs', 'Measured_harmonics_[MPa]_cos', 'Measured_harmonics_[MPa]_sin'))
for item in range(len(tangential_pressure_orders['freq'])):
    the_file.write("{0} {1} {2} {3}\n".format(tangential_pressure_orders['freq'][item], tangential_pressure_orders['abs'][item], tangential_pressure_orders['real'][item], tangential_pressure_orders['imag'][item]))
the_file.close()
print("Data from exported to file {0}".format(the_file_name))


the_file_name = 'my_file3.txt'
the_file = open(the_file_name, 'w')

the_file.write("{0} {1}\n".format('Frequency[Hz]', 'Nauticus_harmonics_[MPa]_abs'))
for item in range(len(nauticus_pressure_orders[0])):
    the_file.write("{0} {1}\n".format(nauticus_pressure_orders[0][item], nauticus_pressure_orders[1][item]))
the_file.close()
print("Data from exported to file {0}".format(the_file_name))


