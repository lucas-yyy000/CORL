import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import qmc
from numba import jit


def generate_radar_config(num_radar_min, num_radar_max, separation_radius=30.0, map_range=500, radar_minimal_separatin_dist=20.0):
    num_radars = np.random.randint(num_radar_min, high=num_radar_max)
    '''
    Poisson disk sampling.
    '''
    rng = np.random.default_rng()
    radius = separation_radius/map_range
    engine = qmc.PoissonDisk(d=2, radius=radius, seed=rng)
    radar_locs = map_range*engine.random(num_radars)

    radar_orientations = (2*np.pi)*(np.random.rand(num_radars))
    valid_radar_locs = []
    valid_radar_orientations = []
    for i in range(radar_locs.shape[0]):
        if np.linalg.norm(radar_locs[i, :]) > radar_minimal_separatin_dist:
            valid_radar_locs.append(radar_locs[i, :])
            valid_radar_orientations.append(radar_orientations[i])
    radar_locs = np.array(valid_radar_locs)
    radar_orientations = np.array(valid_radar_orientations)

    return radar_locs, radar_orientations

def visualiza_radar_config(radar_locs, radar_orientations, radius=30, xlim=None, ylim=None):
    plt.scatter(radar_locs[:, 0], radar_locs[:, 1])
    for i in range(radar_locs.shape[0]):
        plt.arrow(radar_locs[i, 0], radar_locs[i, 1], 10*np.cos(radar_orientations[i]), 10*np.sin(radar_orientations[i]))
        plt.scatter([radar_locs[i, 0] + radius*np.cos(theta) for theta in np.linspace(0, np.pi*2)], [radar_locs[i, 1] + radius*np.sin(theta) for theta in np.linspace(0, np.pi*2)], s=0.5)
    
    if not xlim is None:
        plt.xlim(xlim)
    if not ylim is None:
        plt.ylim(ylim)

@jit(nopython=True)
def get_radar_heat_map(state, radar_locs, img_size, aircraft_detection_range, grid_size, radar_detection_range):
    '''
    state: [x, y, theta]
    '''
    radars_encoding = np.zeros((img_size, img_size))
    theta = state[2]
    loc_to_glob = np.array([[np.cos(theta), -np.sin(theta), state[0]],
                            [np.sin(theta), np.cos(theta), state[1]],
                            [0., 0., 1.]])
    
    glob_to_loc = np.linalg.inv(loc_to_glob)
    # print(glob_to_loc)
    for radar_loc in radar_locs:
        if abs(state[0] - radar_loc[0]) < aircraft_detection_range or abs(state[1] - radar_loc[1]) < aircraft_detection_range:
            glob_loc_hom = np.array([radar_loc[0], radar_loc[1], 1])
            local_loc_hom = np.dot(glob_to_loc, glob_loc_hom)
            radars_loc_coord = local_loc_hom[:2]

            y_grid = np.rint((radars_loc_coord[1]) / grid_size) 
            x_grid = np.rint((radars_loc_coord[0]) / grid_size) 

            for i in range(-int(img_size/2), int(img_size/2)):
                for j in range(-int(img_size/2), int(img_size/2)):
                    radars_encoding[int(i + img_size/2), int(j + img_size/2)] += np.exp(( -(x_grid - i)**2 / (radar_detection_range / grid_size)**2 - (y_grid - j)**2 / (radar_detection_range / grid_size)**2 ))*1e3

    ### Transpose so that x <---> row, y <---> column
    radars_encoding = radars_encoding.T

    ### Make the magnitude correct.
    if np.max(radars_encoding) > 0:
        formatted = (radars_encoding * 255.0 / np.max(radars_encoding)).astype('float32')
    else:
        formatted = radars_encoding.astype('float32')
    
    ### Add one more dimension (batch)
    formatted = formatted[np.newaxis, :, :]

    return formatted