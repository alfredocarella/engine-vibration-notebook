import math
import numpy as np
import pylab
import scipy.fftpack
import scipy.interpolate
import h5py

__author__ = 'alfredoc'


def detect_peak(y_axis, x_axis=None, look_ahead=300, delta=0.0):
    """
    Detects local peaks and valleys in a signal by evaluating every point in a "look_ahead" radius.

    Syntax:
    y_axis -- The signal over which peaks or valleys should be found
    x_axis -- (optional) An x-axis whose values correspond to the y_axis and is used to return the
              position of the peaks. An index of the y_axis is used if omitted.
    lookahead -- (optional) distance to look ahead from a peak candidate to determine if it is a peak.
                 A good tentative value might be: '(sample_rate / period) / f' where '4 >= f >= 1.25'
    delta -- (optional) this specifies a minimum difference between a peak and
             its surroundings in order to consider it a peak.
             A good tentative value might be: delta >= RMS noise * 5.
    return -- two lists [max_peaks, min_peaks] containing the positive and negative peaks respectively.
              Each cell of the lists contains a tuple of: (position, peak_value)
        to get the average peak value do: np.mean(max_peaks, 0)[1] on the
        results to unpack one of the lists into x, y coordinates do:
        x, y = zip(*tab)
    """

    if y_axis is None:
        raise (ValueError, 'Input vector y_axis cannot be "None"')
    length = len(y_axis)

    if x_axis is None and y_axis is not None:
        x_axis = range(length)
    elif length != len(x_axis):
        raise (ValueError, 'Input vectors y_axis and x_axis must have same length')

    if look_ahead < 1:
        raise ValueError("look_ahead must be equal or larger than '1'")

    if not (np.isscalar(delta) and delta >= 0):
        raise ValueError("delta must be a positive number")

    x_axis, y_axis = np.array(x_axis), np.array(y_axis)  # Convert to numpy array if the input is a list
    max_peaks, min_peaks, dump = [], [], []  # Used to pop the first hit which almost always is false

    minimum, maximum = np.Inf, -np.Inf  # maximum and minimum candidates

    # Only detect peak if there is 'look_ahead' amount of points after it
    for index, (x, y) in enumerate(zip(x_axis[:-look_ahead], y_axis[:-look_ahead])):
        if y > maximum:
            maximum, max_pos = y, x
        if y < minimum:
            minimum, min_pos = y, x

        # look for max
        if y < maximum - delta and maximum != np.Inf:
            if y_axis[index: index + look_ahead].max() < maximum:
                max_peaks.append([max_pos, maximum])
                dump.append(True)
                maximum, minimum = np.Inf, np.Inf
                if index + look_ahead >= length:  # end is within look_ahead -> no more peaks
                    break
                continue

        # look for min
        if y > minimum + delta and minimum != -np.Inf:
            if y_axis[index: index + look_ahead].min() > minimum:
                min_peaks.append([min_pos, minimum])
                dump.append(False)
                minimum, maximum = -np.Inf, -np.Inf
                if index + look_ahead >= length:  # end is within look_ahead -> no more valleys
                    break

    # Remove the false hit on the first value of the y_axis
    try:
        if dump[0]:
            max_peaks.pop(0)
        else:
            min_peaks.pop(0)
        del dump
    except IndexError:
        pass

    return [max_peaks, min_peaks]


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


def get_average_pressure_cycle(signal_array, plot_all_cycles=False):
    """
    Receives a vibration signal array and returns the average combustion pressure cycle normalized to degrees.
    """
    signal, pip, time, = signal_array['y'], signal_array['pip'], signal_array['t']
    # Get the time indices for a full 4-stroke cycle
    zero_angle_indices, dummy_var = zip(*detect_peak(pip, look_ahead=1000, delta=2e0)[0])
    full_cycle_indices = zero_angle_indices[0::2]

    # Normalize each cycle to 720 degrees, interpolate to a common resolution and compute an average cycle
    number_of_samples = 8192
    average_pressure = np.zeros(number_of_samples)
    number_of_cycles = len(full_cycle_indices) - 1
    for cycle_idx in range(number_of_cycles):
        cycle_indices = np.arange(full_cycle_indices[cycle_idx], full_cycle_indices[cycle_idx + 1], dtype=int)
        cycle_time = time[cycle_indices] - time[cycle_indices[0]]  # Normalizing so t_0 = 0 for this cycle
        cycle_angle = cycle_time * (720.0 / cycle_time[-1])
        cycle_pressure = scipy.interpolate.interp1d(cycle_angle, signal[cycle_indices])
        cycle_angle = np.linspace(0, 720, num=number_of_samples, endpoint=False)
        single_cycle = cycle_pressure(cycle_angle)
        average_pressure += single_cycle
    average_pressure /= number_of_cycles
    average_cycle = np.array(list(zip(cycle_angle, average_pressure)),
                      dtype={'names':['angle', 'pressure'], 'formats': ['float', 'float']})

    # average_cycle = [cycle_angle, average_pressure]

    if plot_all_cycles:
        pylab.figure()
        for cycle_idx in range(number_of_cycles):
            pylab.plot(cycle_angle, single_cycle)
        pylab.suptitle("Combustion cycles before averaging", fontsize=20)
        pylab.xlabel('Angle ref. pip signal [deg]')
        pylab.ylabel('{0}{1}'.format('Pressure', ' [MPa]'))
        pylab.xlim([0, 720])
    else:
        pass

    return average_cycle


def apply_experimental_setup_corrections(measured_signal, offset_deg, pressure_offset):
    angle = measured_signal['angle']
    pressure = measured_signal['pressure']
    offset_idx = np.where(np.diff(np.sign(angle - offset_deg)))[0]
    shifted_pressure = pressure[range(len(pressure)) - offset_idx] + pressure_offset
    corrected_signal = np.array(list(zip(angle, shifted_pressure)),
                      dtype={'names':['angle', 'pressure'], 'formats': ['float', 'float']})
    return corrected_signal


def get_tangential_pressure(cylinder_pressure_signal, conrod_ratio):
    # This function makes use of kinematic considerations for converting raw pressure data into tangential pressure
    # Tangential Pressure = Cylinder Pressure * SIN(Crank Angle) * { 1 + lambda * COS(Crank Angle) / SQRT[1 - {lambda * COS(Crank Angle)}**2] }
    # where  = Crank Radius / Connecting Rod Length

    crank_angle = np.radians(cylinder_pressure_signal['angle'])
    cylinder_pressure = cylinder_pressure_signal['pressure']

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

    if plot_excitation_orders:
        pylab.figure(3)
        pylab.plot(short_excitation_fft[0], abs(short_excitation_fft[1]))
        pylab.plot(excitation_peaks['freq'], excitation_peaks['abs'], '.g')  # Green dots on excitation peaks

    return excitation_peaks


###############################################
# ######### IMPORT AND PROCESS DATA ######### #
###############################################

# 0- IMPORT VIBRATION SIGNAL FROM h5f FILE
file_name = 'sample_data.h5'
data = {}
with h5py.File(file_name, 'r') as h5f:
    for field in list(h5f):
        data[field] = h5f[field].value

vib_signal = np.array(list(zip(data['time'], data['vibration_amplitude'], data['pip'])),
                      dtype={'names':['t', 'y', 'pip'], 'formats': ['float', 'float', 'float']})

# 1- NORMALIZE TO 720 DEGREES, INTERPOLATE TO A COMMON RESOLUTION AND GET AVERAGE CYCLE
average_pressure_cycle = get_average_pressure_cycle(vib_signal)

# 2- SHIFT SIGNAL REFERENCE TO "Top dead center"=0. ADD AN OFFSET TO GET ZERO PRESSURE AT 540deg
angle_tdc = 58.8  # Angle in degrees
pressure_offset = 0.270 + 0.935  # = 1.205 [MPa]
# pressure_offset = 2.360 + 0.935  #  = 3.295 [MPa]
corrected_pressure_cycle = apply_experimental_setup_corrections(average_pressure_cycle, angle_tdc, pressure_offset)

# 3- CONVERTING TO 'TANGENTIAL PRESSURE'
crank_throw, connecting_rod_length = 0.1075, 0.3925  # [m]
conrod_ratio = crank_throw / connecting_rod_length
tangential_pressure, tangential_modulation = get_tangential_pressure(corrected_pressure_cycle, conrod_ratio)

# 4- CONVERTING TO TORQUE
cylinder_diameter = 0.170  # [m]
instant_torque = get_instant_torque(tangential_pressure, cylinder_diameter, crank_throw, number_of_cylinders=16)

# 5- FFT FOR AN INFINITE NUMBER OF CYCLES
tangential_pressure_orders = get_excitation_orders(tangential_pressure, rpm=1500, max_order=9, plot_excitation_orders=False)
torque_orders = get_excitation_orders(instant_torque, rpm=1500, max_order=9, plot_excitation_orders=False)

nauticus_pressure_orders = [25*np.array([0.5, 1.00, 1.5000, 2.000, 2.5000, 3.5000, 4.0000, 4.5000, 5.0000, 5.5000, 6.0000, 6.5000, 7.0000, 7.5000, 8.0000, 8.5000]),
                            np.array([0.795, 1.345, 1.0451, 0.977, 0.8191, 0.5651, 0.4221, 0.3561, 0.2861, 0.2221, 0.1731, 0.1313, 0.0996, 0.0769, 0.0588, 0.0434])]


###############################################
# ############### PRINT OUTPUT ############## #
###############################################
test_figure = pylab.figure(num=1)

pylab.subplot(3, 1, 1)
# pylab.plot(one_pressure_cycle[0], one_pressure_cycle[1], label='Average cycle')
pylab.plot(corrected_pressure_cycle[0], corrected_pressure_cycle[1], 'b', label='Average cycle (tdc=0)')
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


# ###############################################
# # ############### EXPORT DATA ############### #
# ###############################################
# the_file_name = 'my_file1.txt'
# the_file = open(the_file_name, 'w')
#
# the_file.write("{0} {1} {2} {3}\n".format('Angle[deg]', 'Tangential_modulation', 'Pressure_cycle_[MPa]', 'Tangential_pressure[MPa]'))
# for item in range(len(tangential_modulation[0])):
#     the_file.write("{0} {1} {2} {3}\n".format(tangential_modulation[0][item], tangential_modulation[1][item], corrected_pressure_cycle[1][item], tangential_pressure[1][item]))
# the_file.close()
# print("Data from exported to file {0}".format(the_file_name))
#
#
# the_file_name = 'my_file2.txt'
# the_file = open(the_file_name, 'w')
#
# the_file.write("{0} {1} {2} {3}\n".format('Frequency[Hz]', 'Measured_harmonics_[MPa]_abs', 'Measured_harmonics_[MPa]_cos', 'Measured_harmonics_[MPa]_sin'))
# for item in range(len(tangential_pressure_orders['freq'])):
#     the_file.write("{0} {1} {2} {3}\n".format(tangential_pressure_orders['freq'][item], tangential_pressure_orders['abs'][item], tangential_pressure_orders['real'][item], tangential_pressure_orders['imag'][item]))
# the_file.close()
# print("Data from exported to file {0}".format(the_file_name))
#
#
# the_file_name = 'my_file3.txt'
# the_file = open(the_file_name, 'w')
#
# the_file.write("{0} {1}\n".format('Frequency[Hz]', 'Nauticus_harmonics_[MPa]_abs'))
# for item in range(len(nauticus_pressure_orders[0])):
#     the_file.write("{0} {1}\n".format(nauticus_pressure_orders[0][item], nauticus_pressure_orders[1][item]))
# the_file.close()
# print("Data from exported to file {0}".format(the_file_name))
#
#
