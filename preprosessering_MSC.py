import spectral as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import pandas as pd


def sample_mean_spectra(directory, side, brick_nr):

    # Load image
    hdr = sp.envi.open(directory)
    wvl = hdr.bands.centers
    rows, cols, bands = hdr.nrows, hdr.ncols, hdr.nbands
    meta = hdr.metadata
    img = hdr.load()

    # White reference
    means = np.empty([cols, bands])

    for j in range(rows):
        if img[j, 0, 0] < 0.1:
            transition = j + 40  # -10
            break

    for i in range(cols):
        for band in range(bands):
            means[i, band] = np.mean(img[:transition, i, band])

    corr_img = img.copy()
    for i in range(cols):
        for band in range(bands):
            corr_img[:, i, band] = (corr_img[:, i, band] / means[i][band]) * 0.6


    # Crop the image
    for j in range(rows):
        # if img[j,0,0] < 0.1:
        if corr_img[j, 0, 0] < 0.5 * 0.6:
            first_dark = j + 10
            break

    for j in range(rows):
        if (j > first_dark + 700) & (img[j, 0, 0] < 0.09 * 0.6):  # 800??? funker for alle??
            last_dark = j + 10
            break

    if side == "A" or side == "B":
        crop_img = corr_img[first_dark + 50:last_dark - 50, :, :]

    if side == "C" or side == "D":
        crop_img = corr_img[first_dark:last_dark, 50:260, :]


    # Smoothing filter
    savgol_img = savgol_filter(crop_img, 5, 1)


    # Absorbans
    x, y, z = savgol_img.shape
    abs_img = savgol_img.copy()
    abs_img = np.log10(1 / savgol_img)

    # Without SNV
    snv_img = abs_img.copy()
    x, y, z = abs_img.shape

    # Calculate dark filter
    x, y, z = snv_img.shape

    # Limit for bricks when not dried:
    # if brick_nr >= 12.:
    #   limit = 0.1
    # else:
    #   limit = 0.5 * 0.6

    # Limit for bricks when dried:
    limit = 0.8 * 0.6

    dark_filter = []
    for j in range(x):
        filter_rows = []
        for i in range(y):
            if crop_img[j, i, 270] < limit:
                filter_rows.append(False)
            else:
                filter_rows.append(True)
        dark_filter.append(filter_rows)
    dark_filter = np.array(dark_filter)

    plt.figure(figsize=(5, 5))
    plt.imshow(dark_filter) #Print dark filter to check it looks fine
    plt.show()

    # Find mean spectra in 12 grid
    x, y, z = crop_img.shape
    height = 5
    width = 4

    height_interval = np.linspace(0, x, height)
    width_interval = np.linspace(0, y, width)

    mean_vals_grid = []
    counter = 1

    plt.figure(figsize=(10, 10))

    for j in range(len(height_interval) - 1):
        for i in range(len(width_interval) - 1):
            roi = snv_img[round(height_interval[j]):round(height_interval[j + 1]),
                  round(width_interval[i]):round(width_interval[i + 1]), :]  # region of interest
            roi_filter = dark_filter[round(height_interval[j]):round(height_interval[j + 1]),
                         round(width_interval[i]):round(width_interval[i + 1])]

            mean_vals = np.zeros(bands)

            for band in range(bands):
                mean_vals[band] = np.mean(roi[:, :, band][roi_filter])
            mean_vals_grid.append(mean_vals)
            plt.plot(wvl, mean_vals, label=f'{counter}')
            counter += 1

    plt.legend()
    plt.savefig(f'Murstein{brick_nr}_dry_grid_spectra_B_KUTTET.pdf') # Save figure with 12 spectra per brick
    plt.show()

    return mean_vals_grid


# Code for MSC found at: https://towardsdatascience.com/scatter-correction-and-outlier-detection-in-nir-spectroscopy-7ec924af668

def msc(input_data, reference=None):
    """
        :msc: Scatter Correction technique performed with mean of the sample data as the reference.        :param input_data: Array of spectral data
        :type input_data: DataFrame        :returns: data_msc (ndarray): Scatter corrected spectra data
    """
    eps = np.finfo(np.float32).eps
    input_data = np.array(input_data, dtype=np.float64)
    ref = []
    sampleCount = int(len(input_data))

    # mean centre correction
    for i in range(input_data.shape[0]):
        input_data[i, :] -= input_data[i, :].mean()

    # Get the reference spectrum. If not given, estimate it from the mean    # Define a new array and populate it with the corrected data
    data_msc = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        for j in range(0, sampleCount, 10):
            ref.append(np.mean(input_data[j:j + 10], axis=0))
            # Run regression
            fit = np.polyfit(ref[i], input_data[i, :], 1, full=True)
            # Apply correction
            data_msc[i, :] = (input_data[i, :] - fit[0][1]) / fit[0][0]

    return (data_msc)


# Here the functions above are used, first to preprosess the images without MSC, and then MSC is performed on the spectra.

def get_MSC_prepross(bricks_nr, sides, directories):

    # First all but MSC preprosessing is used and stored in a dataframe
    Represent_grid = []

    for directory, brick_nr, side in zip(directories, bricks_nr, sides):
        mean_vals = sample_mean_spectra(directory, side, brick_nr)

        mean_vals_df = pd.DataFrame(mean_vals)
        brick_nr = np.full((12), brick_nr)
        mean_vals_df["Murstein nr"] = brick_nr
        Represent_grid.append(mean_vals_df)

    Represent_grid = pd.DataFrame()
    Represent_grid = Represent_grid.append([Represent_grid[0], Represent_grid[1], Represent_grid[2], Represent_grid[3],
                                            Represent_grid[4], Represent_grid[5], Represent_grid[6], Represent_grid[7],
                                            Represent_grid[8], Represent_grid[9], Represent_grid[10], Represent_grid[11],
                                            Represent_grid[12], Represent_grid[13], Represent_grid[14], Represent_grid[15],
                                            Represent_grid[16]], ignore_index=True)

    # MSC is preformed on all the spectras. The mean spectra is calculated for each side.
    MSC_prepross_data = []
    for i in range(len(directories)):
        MSC_data = msc(list(Represent_grid[i].iloc[:, :-1].values), reference=None)
        MSC_data_df = pd.DataFrame(MSC_data)

        brick_nr = np.full((12), float(i + 1))
        brick_side = np.full((12), side)
        MSC_data_df["Murstein nr"] = brick_nr
        MSC_data_df["Side"] = brick_side

        MSC_prepross_data.append(MSC_data_df)

    MSC_prepross_data_df = pd.DataFrame()
    MSC_prepross_data_df = MSC_prepross_data_df.append([MSC_prepross_data[0], MSC_prepross_data[1], MSC_prepross_data[2],
                                                        MSC_prepross_data[3], MSC_prepross_data[4], MSC_prepross_data[5],
                                                        MSC_prepross_data[6], MSC_prepross_data[7], MSC_prepross_data[8],
                                                        MSC_prepross_data[9], MSC_prepross_data[10], MSC_prepross_data[11],
                                                        MSC_prepross_data[12], MSC_prepross_data[13], MSC_prepross_data[14],
                                                        MSC_prepross_data[15], MSC_prepross_data[16]], ignore_index=True)

    return MSC_prepross_data_df
