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

    # SNV
    snv_img = abs_img.copy()
    x, y, z = abs_img.shape

    for i in range(y):
        for j in range(x):
            snv_img[j, i, :] = (snv_img[j, i, :] - np.mean(snv_img[j, i, :])) / (np.std(snv_img[j, i, :]))

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