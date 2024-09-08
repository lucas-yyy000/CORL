import numpy as np
from scipy.stats import bernoulli

from evasion_guidance.scripts.evasion_risk import EvasionRisk

class DetectionSimulator():
    
    def __init__(self, radar_locations, radar_detection_range, interceptor_launch_time, interceptor_abort_time):
        super().__init__()
        self.radar_locations = radar_locations
        self.num_radars = len(self.radar_locations)
        self.radar_tracking_duration = np.zeros(self.num_radars)
        self.radar_tracking_lost_duration = np.zeros(self.num_radars)
        self.radar_tracking_lost_prev_step = np.zeros(self.num_radars)
        
        self.radar_detection_range = radar_detection_range
        self.risk_evaluator = EvasionRisk(self.radar_locations, 0.0, self.radar_detection_range)
        
        # Maximum number of time steps for the agent to be detected.
        self.interceptor_launch_time = interceptor_launch_time

        # Minimum number of time steps for the interceptpr to abort.
        self.interceptor_abort_time = interceptor_abort_time

    def update_shutdown(self, state, u):
        '''
        Update state of the env.
        radar_tracking_duration records the total duration of the aircraft being tracked by each radar
        radar_tracking_lost_duration records how many steps has each radar lost track of the aircraft
        radar_tracking_lost_prev_step records whether each radar lost track of the aircraft during the previous time step
        '''
        p_detection, instant_prob = self.risk_evaluator.evalute_risk(state, u, return_list=True)
        radar_detection_result = bernoulli.rvs(instant_prob)
        # self.radar_tracking_time += radar_detection_result
        for i in range(self.num_radars):
            # If lost track for interceptor_abort_time, abort.
            if self.radar_tracking_duration[i] == 0.:
                self.radar_tracking_duration[i] += radar_detection_result[i]
                self.radar_tracking_lost_duration[i] = 0.0
                self.radar_tracking_lost_prev_step[i] = 0.0
                continue

            if self.radar_tracking_duration[i] > 0 and radar_detection_result[i] == 0:
                if self.radar_tracking_lost_prev_step[i] == 0.0 and self.radar_tracking_lost_duration[i] == 0.0:
                    self.radar_tracking_lost_duration[i] = 1.0
                elif self.radar_tracking_lost_prev_step[i] > 0.0:
                    self.radar_tracking_lost_duration[i] += 1
                    
                if self.radar_tracking_lost_duration[i] >= self.interceptor_abort_time:
                    self.radar_tracking_duration[i] = 0
                    self.radar_tracking_lost_duration[i] = 0
                    self.radar_tracking_lost_prev_step[i] = 0
                    continue

                self.radar_tracking_lost_prev_step[i] = 1.0
                self.radar_tracking_duration[i] += 1.0

            if self.radar_tracking_duration[i] > 0 and radar_detection_result[i] > 0.:
                    self.radar_tracking_duration[i] += 1.0
                    self.radar_tracking_lost_duration[i] = 0
                    self.radar_tracking_lost_prev_step[i] = 0

            if self.radar_tracking_duration[i] >= self.interceptor_launch_time:
                return True
            
        return False