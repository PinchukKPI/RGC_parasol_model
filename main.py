
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.io import loadmat


def load_white_noise_data(cell_id):
    data = loadmat('Data/elife-38841-fig4-data1-v2.mat')  # loadmat is a function in scipy.io
    # data from Figure 4 can be downloaded from https://doi.org/10.7554/eLife.38841.013
    # "Receptive field center-surround interactions mediate context-dependent spatial contrast encoding in the retina"
    # Maxwell H Turner, Gregory W Schwartz, Fred Rieke  DOI: https://doi.org/10.7554/eLife.38841

    # Stimuli and corresponding responses of On and Off parasol RGCs to center-surround
    # white noise stimulation have been concatenated into vector arrays.
    # Note that the data were collected in interleaved trials.
    # This data includes excitatory conductance responses (in nS) that were
    # estimated using measured excitatory current responses and an estimate
    # of the excitatory driving force for each cell. Data are sampled at
    # 10 Khz = 1e4 Hz  so  8000 items with rate 1e4 Hz gives 0.8 sec
    # 1 [nS - nanoSiemens] is equal to 10e-9 [A/V - Amper/Volt]

    # CenterSurroundWhiteNoise is a cell array, each entry of which is a structure that corresponds
    # to a cell in the dataset. Structure fields:
    # .stimulus = concatenated stimulus traces for .center and .surround stimuli
    # .response = concatenated excitatory conductance response traces (in nS)
    # for .center, .surround and .centerSurround stimuli
    # Note that data are concatenated and grouped for convenience, but were acquired in interleaved trials

    center_surround_white_noise = data['CenterSurroundWhiteNoise']  # .item extracts a scalar value
    cell_data = center_surround_white_noise[0, cell_id]
    stimulus = cell_data[0, 0]['stimulus']
    response = cell_data[0, 0]['response']
    # Extract the stimulus intensity
    s_cntr = stimulus['center'][0][0][0]
    s_srnd = stimulus['surround'][0][0][0]
    r_cntr = response['center'][0][0][0]
    r_srnd = response['surround'][0][0][0]
    r_cntr_srnd = response['centerSurround'][0][0][0]

    return s_cntr, s_srnd, r_cntr, r_srnd, r_cntr_srnd


def load_natural_data(cell_id):
    data = loadmat('Data/elife-38841-fig7-data1-v2.mat') # data from Figure 7
    # downloaded from https://doi.org/10.7554/eLife.38841.017
    # The structure contains excitatory current responses (baseline subtracted, in pA) of On and Off parasol RGCs
    # to center-surround naturalistic luminance stimuli. Data are sampled at 10 Khz.
    # In the experiments in Figure 7, we updated the intensity of a disc (annulus) in the center (surround)
    # every 200 ms, which is consistent with typical human fixation periods during free-viewing,
    # although less than the mean period (for efficient data collection).
    # To compute natural intensity stimuli for the center and surround,
    # we selected many random patches from a natural image and measured the mean intensity
    # within a circle of diameter 200 μm (for the RF center) and within an annulus
    # with inner and outer diameters 200 and 600 μm, respectively (for the RF surround).

    center_surround_natural_image_liminance = data['CenterSurroundNaturalImageLuminance']  # .item extracts a scalar value
    cell_data = center_surround_natural_image_liminance[0, cell_id]
    stimulus = cell_data[0, 0]['stimulus']
    response = cell_data[0, 0]['response']
    s_cntr = stimulus['controlCenter'][0][0][0]
    s_srnd = stimulus['controlSurround'][0][0][0]
    s_srnd_shffl = stimulus['shuffleSurround'][0][0][0]

    r_cntr = response['controlCenter'][0][0][0]  # only s_cntr
    r_srnd = response['controlSurround'][0][0][0]  # only s_srnd
    r_cntr_srnd = response['controlCenterSurround'][0][0][0]  # s_srnd + s_srnd

    r_cntr_shffl = response['shuffleCenter'][0][0][0]  # only s_cntr
    r_srnd_shffl = response['shuffleSurround'][0][0][0]  # only s_srnd_shffl
    r_cntr_srnd_shffl = response['shuffleCenterSurround'][0][0][0]  # s_srnd + s_srnd_shffl

    return s_cntr, s_srnd, s_srnd_shffl, r_cntr, r_srnd, r_cntr_srnd, r_cntr_shffl, r_srnd_shffl, r_cntr_srnd_shffl


def filter_and_spikes_detect(response):
    # hi-pass filtering
    kernel = [-0.1, -0.1, -0.1, -0.1, -0.1,  1,  -0.1, -0.1, -0.1, -0.1, -0.1 ]  # sum should be 0
    kernel_size = 11
    response_size = response.size
    filtered = np.zeros(response_size)
    for i in range(response_size):
        kernel_sum = 0
        for filter_index in range(6, kernel_size-7):
            #if 0 < (i + filter_index - 6) < response_size:
            kernel_sum = kernel_sum + kernel[filter_index] * response[i + filter_index - 6]
        filtered[i] = kernel_sum

    # detect spikes
    threshold = 0.9
    spikes = np.zeros(response_size)
    for i in range(response_size - 1):
        if filtered[i] <= threshold < filtered[i + 1]:
            spikes[i] = 1

    return spikes

if __name__ == '__main__':
    cell = 13  # 0-7 Off-center   8-14 On-center
    #stimulus_c, stimulus_s, stimulus_s_shffl, response_c, response_s, response_cs, \
    #response_c_shffl, response_s_shffl, response_cs_shffl = load_natural_data(cell)

    stimulus_c, stimulus_s, response_c, response_s, response_cs = load_white_noise_data(cell)
    spikes_c = filter_and_spikes_detect(response_c)
  #  spikes_s = filter_and_spikes_detect(response_s)
  #  spikes_cs = filter_and_spikes_detect(response_cs)



