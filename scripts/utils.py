import numpy as np

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