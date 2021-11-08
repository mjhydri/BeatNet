import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng

rng = default_rng()
# using bar pointer state space implemented in Madmom
from madmom.features.beats_hmm import BarStateSpace, BarTransitionModel
from madmom.ml.hmm import TransitionModel, ObservationModel


class BDObservationModel(ObservationModel):
    """
    Observation model for beat and downbeat tracking with particle filtering.

    Parameters
    ----------
    state_space : :class:`BarStateSpace` instance
        BarStateSpace instance.
    observation_lambda : str
        Based on the first character of this parameter, each (down-)beat period gets split into (down-)beat states
        "B" stands for border model which classifies 1/(observation lambda) fraction of states as downbeat states and
        the rest as the beat states (if it is used for downbeat tracking state space) or the same fraction of states
        as beat states and the rest as the none beat states (if it is used for beat tracking state space).
        "N" model assigns a constant number of the beginning states as downbeat states and the rest as beat states
         or beginning states as beat and the rest as none-beat states
        "G" model is a smooth Gaussian transition (soft border) between downbeat/beat or beat/none-beat states

    """

    def __init__(self, state_space, observation_lambda):

        if observation_lambda[0] == 'B':
            observation_lambda = int(observation_lambda[1:])
            # compute observation pointers
            # always point to the non-beat densities
            pointers = np.zeros(state_space.num_states, dtype=np.uint32)
            # unless they are in the beat range of the state space
            border = 1. / observation_lambda
            pointers[state_space.state_positions % 1 < border] = 1
            # the downbeat (i.e. the first beat range) points to density column 2
            pointers[state_space.state_positions < border] = 2
            # instantiate a ObservationModel with the pointers
            super(BDObservationModel, self).__init__(pointers)

        elif observation_lambda[0] == 'N':
            observation_lambda = int(observation_lambda[1:])
            # compute observation pointers
            # always point to the non-beat densities
            pointers = np.zeros(state_space.num_states, dtype=np.uint32)
            # unless they are in the beat range of the state space
            for i in range(observation_lambda):
                border = np.asarray(state_space.first_states) + i
                pointers[border[1:]] = 1
                # the downbeat (i.e. the first beat range) points to density column 2
                pointers[border[0]] = 2
                # instantiate a ObservationModel with the pointers
            super(BDObservationModel, self).__init__(pointers)

        elif observation_lambda[0] == 'G':
            observation_lambda = float(observation_lambda[1:])
            pointers = np.zeros((state_space.num_beats + 1, state_space.num_states))
            for i in range(state_space.num_beats + 1):
                pointers[i] = gaussian(state_space.state_positions, i, observation_lambda)
            pointers[0] = pointers[0] + pointers[-1]
            pointers[1] = np.sum(pointers[1:-1], axis=0)
            pointers = pointers[:2]
            super(BDObservationModel, self).__init__(pointers)


def gaussian(x, mu, sig):
    return np.exp(-np.power((x - mu) / sig, 2.) / 2)  # /(np.sqrt(2.*np.pi)*sig)


#   assigning beat vs non-beat weights
def beat_densities(observations, observation_model, state_model):
    new_obs = np.zeros(state_model.num_states, float)
    if len(np.shape(observation_model.pointers)) != 2:  # B or N
        new_obs[np.argwhere(observation_model.pointers == 2)] = observations
        new_obs[np.argwhere(observation_model.pointers == 0)] = 0.03
    elif len(np.shape(observation_model.pointers)) == 2:  # G
        new_obs = observation_model.pointers[0] * observations
        new_obs[new_obs < 0.005] = 0.03
    return new_obs

#   assigning downbeat vs beat weights
def down_densities(observations, observation_model, state_model):
    new_obs = np.zeros(state_model.num_states, float)
    if len(np.shape(observation_model.pointers)) != 2:  # B or N
        new_obs[np.argwhere(
            observation_model.pointers == 2)] = observations[1]
        new_obs[np.argwhere(
            observation_model.pointers == 0)] = observations[0]
    elif len(np.shape(observation_model.pointers)) == 2:  # G
        new_obs = observation_model.pointers[0] * observations
        new_obs[new_obs < 0.005] = 0.03
    return new_obs

#   assigning downbeat vs beat weights - second model
def down_densities2(observations, beats_per_bar):
    new_obs = np.zeros(beats_per_bar, float)
    new_obs[0] = observations[1]  # downbeat activation
    new_obs[1:] = observations[0]  # beat activation
    return new_obs

#   Inference initialization
class particle_filter_cascade:
    np.random.seed(1)
    PARTICLE_SIZE = 1500
    DOWN_PARTICLE_SIZE = 250
    MIN_BPM = 55.
    MAX_BPM = 215.  # 215.
    NUM_TEMPI = 300
    LAMBDA_B = 60  # beat transition lambda
    LAMBDA_D = 0.1  # downbeat transition lambda
    OBSERVATION_LAMBDA_B = "B56"  # beat observation lambda
    OBSERVATION_LAMBDA_D = "B56"  # downbeat observation lambda
    fps = 50
    T = 1 / fps
    MIN_BEAT_PER_BAR = 2
    MAX_BEAT_PER_BAR = 6
    OFFSET = 4  # The point of time after which the inference model starts to work. Can be zero!
    IG_THRESHOLD = 0.4  # Information Gate threshold

    def __init__(self, beats_per_bar=[], particle_size=PARTICLE_SIZE, down_particle_size=DOWN_PARTICLE_SIZE,
                 min_bpm=MIN_BPM, max_bpm=MAX_BPM, num_tempi=NUM_TEMPI, min_beats_per_bar=MIN_BEAT_PER_BAR,
                 max_beats_per_bar=MAX_BEAT_PER_BAR, offset=OFFSET, ig_threshold=IG_THRESHOLD, lambda_b=LAMBDA_B,
                 lambda_d=LAMBDA_D, observation_lambda_b=OBSERVATION_LAMBDA_B, observation_lambda_d=OBSERVATION_LAMBDA_D,
                 fps=None, plot=False, **kwargs):
        self.particle_size = particle_size
        self.down_particle_size = down_particle_size
        self.particle_filter = []
        self.beats_per_bar = beats_per_bar
        self.fps = fps
        self.Lambda_b = lambda_b
        self.Lambda_d = lambda_d
        self.observation_lambda_b = observation_lambda_b
        self.observation_lambda_d = observation_lambda_d
        self.plot = plot
        self.min_beats_per_bar = min_beats_per_bar
        self.max_beats_per_bar = max_beats_per_bar
        self.offset = offset
        self.ig_threshold = ig_threshold
        # convert timing information to construct a beat state space
        min_interval = 60. * fps / max_bpm
        max_interval = 60. * fps / min_bpm
        self.st = BarStateSpace(1, min_interval, max_interval, num_tempi)    # beat tracking state space
        if beats_per_bar:   # if the number of beats per bar is given
            self.st2 = BarStateSpace(1, min(self.beats_per_bar ), max(self.beats_per_bar),
                                max(self.beats_per_bar ) - min(self.beats_per_bar) + 1)   # downbeat tracking state space
        else:   # if the number of beats per bar is not given
            self.st2 = BarStateSpace(1, self.min_beats_per_bar, self.max_beats_per_bar, 5)  # downbeat tracking state space
        tm = BarTransitionModel(self.st, self.Lambda_b)
        self.tm = list(TransitionModel.make_dense(tm.states, tm.pointers, tm.probabilities))   # beat transition model
        self.om = BDObservationModel(self.st, self.observation_lambda_b)   # beat observation model
        self.st.last_states = list(np.concatenate(self.st.last_states).flat)    # beat last states
        self.om2 = BDObservationModel(self.st2, self.observation_lambda_d)  # downbeat observation model
        self.tm2 = np.zeros((len(self.st2.first_states[0]), len(self.st2.first_states[0])))  # downbeat transition model
        for i in range(len(self.st2.first_states[0])):
            for j in range(len(self.st2.first_states[0])):
                if i == j:
                    self.tm2[i, j] = 1 - self.Lambda_d
                else:
                    self.tm2[i, j] = self.Lambda_d / (len(self.st2.first_states[0]) - 1)
        pass

    def process(self, activations):

        """
        Running Particle filtering over the given activation function to infer beats/downbeats.

        Parameters
        ----------
        activations : numpy array, shape (num_frames, 2)
            Activation function with probabilities corresponding to beats
            and downbeats given in the first and second column, respectively.

        Returns
        -------
        beats, downbeats : numpy array, shape (num_beats, 2)
            Detected (down-)beat positions [seconds] and beat numbers.

        """

        T = 1 / self.fps
        counter = 0
        if self.plot:
            fig = plt.figure(figsize=(1800 / 96, 900 / 96), dpi=96)
            subplot1 = fig.add_subplot(211)
            subplot2 = fig.add_subplot(212)
            # line1 = subplot1.scatter(meter.st.state_positions,np.max(meter.st.state_intervals)-meter.st.state_intervals, marker='o', color='grey', alpha=0.08)
            # line2 = subplot2.scatter(self.st2.state_positions, np.max(self.st2.state_intervals) - self.st2.state_intervals, marker='o', color='grey', alpha=0.08)

        path = np.zeros((1, 2), dtype=float)
        position = []
        #   particles initialization
        particles = np.sort(
            np.random.choice(np.arange(0, self.st.num_states - 1), self.particle_size, replace=True))
        down_particles = np.sort(
            np.random.choice(np.arange(0, self.st2.num_states - 1), self.down_particle_size, replace=True))
        #   applying the offset and information gate thresholds
        beat = np.squeeze(self.st.first_states)
        activations = activations[int(self.offset / T):]
        both_activations = activations.copy()
        activations = np.max(activations, axis=1)
        activations[activations < self.ig_threshold] = 0.03

        for i in range(len(activations)):  # loop through all frames to infer beats/downbeats
            counter += 1
            if self.plot:  # activate this when you want to plot the performance
                position = np.median(self.st.state_positions[particles])    # calculating beat clutter position
            gathering = int(np.median(particles))   # calculating beat particles clutter
            # checking if the clutter is within the beat interval
            if ((gathering - beat[self.st.state_intervals[beat] == self.st.state_intervals[gathering]]) < (
                    int(.07 / T)) + 1).any() and (self.offset + counter * T) - path[-1][0] > .4 * T * \
                    self.st.state_intervals[
                        gathering]:
                last1 = down_particles[np.in1d(down_particles, self.st2.last_states)]   # downbeat motion
                state1 = down_particles[~np.in1d(down_particles, self.st2.last_states)] + 1
                for j in range(len(last1)):
                    arg1 = np.argwhere(self.st2.last_states[0] == last1[j])[0][0]
                    nn = np.random.choice(self.st2.first_states[0], 1, p=(np.squeeze(self.tm2[arg1])))
                    state1 = np.append(state1, nn)
                down_particles = state1
                obs2 = down_densities(both_activations[i], self.om2, self.st2)
                down_particles = universal_resample(down_particles, obs2[down_particles])  # downbeat correction
                m = np.bincount(down_particles)
                down_max = np.argmax(m)  # calculating downbeat particles clutter
                if down_max in self.st2.first_states[0]:
                    path = np.append(path, [[self.offset + counter * T, 1]], axis=0)
                else:
                    path = np.append(path, [[self.offset + counter * T, 2]], axis=0)
                if self.plot:
                    m1 = np.r_[True, down_particles[:-1] != down_particles[1:], True]
                    counts1 = np.diff(np.flatnonzero(m1))
                    unq1 = down_particles[m1[:-1]]
                    part1 = np.c_[unq1, counts1]
                    subplot2.clear()
                    subplot2.scatter(self.st2.state_positions,
                                     np.max(self.st2.state_intervals) - self.st2.state_intervals,
                                     marker='o',
                                     color='grey', alpha=0.2, s=50)

                    subplot2.scatter(self.st2.state_positions[self.om2.pointers == 2],
                                     np.max(self.st2.state_intervals) - self.st2.state_intervals[
                                         self.om2.pointers == 2],
                                     marker='o',
                                     color='green', s=50, alpha=both_activations[i][1])
                    subplot2.scatter(self.st2.state_positions[self.om2.pointers != 2],
                                     np.max(self.st2.state_intervals) - self.st2.state_intervals[
                                         self.om2.pointers != 2],
                                     marker='o',
                                     color='orange', s=50, alpha=both_activations[i][0])
                    subplot2.scatter(self.st2.state_positions[part1[:, 0]],
                                     np.max(self.st2.state_intervals) - self.st2.state_intervals[part1[:, 0]],
                                     marker='o',
                                     s=part1[:, 1] * 4, color="red")
                    position2 = np.max(self.st2.state_positions[down_max])
                    subplot2.axvline(x=position2) # calculating downbeat clutter position
            obs = beat_densities(activations[i], self.om, self.st)
            if self.plot:
                if counter % 1 == 0:  # choosing how often to plot
                    m = np.r_[True, particles[:-1] != particles[1:], True]
                    counts = np.diff(np.flatnonzero(m))
                    unq = particles[m[:-1]]
                    part = np.c_[unq, counts]
                    subplot1.scatter(self.st.state_positions,
                                     np.max(self.st.state_intervals) - self.st.state_intervals,
                                     marker='o',
                                     color='grey', alpha=0.2)
                    if both_activations[i][0] > both_activations[i][1]:
                        subplot1.scatter(self.st.state_positions[self.om.pointers == 2],
                                         np.max(self.st.state_intervals) - self.st.state_intervals[
                                             self.om.pointers == 2],
                                         marker='o',
                                         color='yellow', alpha=activations[i])
                    else:
                        subplot1.scatter(self.st.state_positions[self.om.pointers == 2],
                                         np.max(self.st.state_intervals) - self.st.state_intervals[
                                             self.om.pointers == 2],
                                         marker='o',
                                         color='green', alpha=activations[i])
                    plt.xlabel("ϕ", size=20)
                    h = plt.ylabel("ϕ'  ", size=20)
                    h.set_rotation(0)
                    plt.xticks([])
                    plt.yticks([])
                    subplot1.scatter(self.st.state_positions[part[:, 0]],
                                     np.max(self.st.state_intervals) - self.st.state_intervals[part[:, 0]],
                                     marker='o',
                                     s=part[:, 1] * 2, color="red")
                    subplot1.axvline(x=position)
                    plt.pause(0.002)
                    subplot1.clear()

            if activations[i] > 0.1:  # resampling is done only when there is a meaningful activation
                particles = universal_resample(particles, obs[particles], )  # beat correction
            last = particles[np.in1d(particles, self.st.last_states)]   # beat motion
            state = particles[~np.in1d(particles, self.st.last_states)] + 1
            for j in range(len(last)):
                args = np.argwhere(self.tm[1] == last[j])
                probs = self.tm[2][args]
                nn = np.random.choice(np.squeeze(self.tm[0][args]), 1, p=(np.squeeze(probs)))
                state = np.append(state, nn)
            particles = state
        return path[1:]


def universal_resample(particles, weights, ):  # state_space
    new_particles = []
    J = len(particles)
    weights = weights / sum(weights)
    r = np.random.uniform(0, 1 / J)
    i = 0
    c = weights[0]
    for j in range(J):
        U = r + j * (1 / J)
        while U > c:
            i += 1
            c += weights[i]
        new_particles = np.append(new_particles, particles[i])
    new_particles = new_particles.astype(int)
    return new_particles


def systematic_resample(particles, weights):
    N = len(weights)
    # make N subdivisions, choose positions
    # with a consistent random offset
    positions = (np.random.randint(0, N) + np.arange(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N & j < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return particles[indexes]


def stratified_resample(particles, weights):
    N = len(weights)
    # make N subdivisions, chose a random position within each one
    positions = (np.random.randint(0, N) + np.arange(N)) / N

    indexes = np.zeros(N, 'i')
    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < N & j < N:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return particles[indexes]
