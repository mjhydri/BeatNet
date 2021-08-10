

import numpy as np
import matplotlib.pyplot as plt
from numpy.random import default_rng
rng = default_rng()
# state spaces
from madmom.features.beats_hmm import BarStateSpace, BarTransitionModel    # using bar pointer state space implementation form madmom
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
    return np.exp(-np.power((x - mu)/sig, 2.)/2) #/(np.sqrt(2.*np.pi)*sig)

class meter(object):
    def __init__(self, st, tm, tm1, om):
        self.st = st
        self.tm = tm
        self.tm = tm1
        self.om = om


def densities(observations, observation_model, state_model):
    new_obs = np.zeros(state_model.num_states, float)
    if len(np.shape(observation_model.pointers)) != 2:  # B or N
        new_obs[np.argwhere(
            observation_model.pointers == 2)] = observations  # * np.min(state_model.state_intervals) / state_model.state_intervals[np.argwhere(observation_model.pointers == 2)]
        new_obs[np.argwhere(
            observation_model.pointers == 0)] = 0.03  # ((1-alpha) * densities[2] * np.min(state_model.state_intervals) / state_model.state_intervals[np.argwhere(observation_model.pointers == 2)])
    elif len(np.shape(observation_model.pointers)) == 2:  # G
        new_obs = observation_model.pointers[
                      0] * observations  # observation_model.pointers[0] = downbeat weigths   observation_model.pointers[1] = beat weigths
        new_obs[new_obs < 0.005] = 0.03
    return new_obs


def densities2(observations, observation_model, state_model):
    new_obs = np.zeros(state_model.num_states, float)

    if len(np.shape(observation_model.pointers)) != 2:  # B or N
        new_obs[np.argwhere(
            observation_model.pointers == 2)] = observations[1]  # * np.min(state_model.state_intervals) / state_model.state_intervals[np.argwhere(observation_model.pointers == 2)]
        new_obs[np.argwhere(
            observation_model.pointers == 0)] = observations[0]   # ((1-alpha) * densities[2] * np.min(state_model.state_intervals) / state_model.state_intervals[np.argwhere(observation_model.pointers == 2)])
    elif len(np.shape(observation_model.pointers)) == 2:  # G
        new_obs = observation_model.pointers[
                      0] * observations  # observation_model.pointers[0] = downbeat weigths   observation_model.pointers[1] = beat weigths
        new_obs[new_obs < 0.005] = 0.03

    # return the densities
    return new_obs

def densities_down(observations, beats_per_bar):
    new_obs = np.zeros(beats_per_bar, float)
    new_obs[0] = observations[1]  #downbeat
    new_obs[1:] = observations[0]  #beat
    return new_obs

class particle_filter_cascade:

    np.random.seed(1)
    PARTICLE_SIZE = 1500
    DOWN_PARTICLE_SIZE = 250
    STATE_FLAG = 0
    MIN_BPM = 55.
    MAX_BPM = 215.  # 215.
    NUM_TEMPI = 300
    LAMBDA_B = 60    # beat transition lambda
    Lambda2 = 0.1   # downbeat transition lambda
    OBSERVATION_LAMBDA = "B56"
    THRESHOLD = None
    fps = 50
    T = 1 / fps


    def __init__(self, beats_per_bar=[], particle_size=PARTICLE_SIZE, down_particle_size=DOWN_PARTICLE_SIZE,
                 state_flag=STATE_FLAG, min_bpm=MIN_BPM, max_bpm=MAX_BPM, num_tempi=NUM_TEMPI,
                 lambda_b=LAMBDA_B, lambda2=Lambda2, observation_lambda=OBSERVATION_LAMBDA,
                 threshold=THRESHOLD, fps=None, plot=False, **kwargs):
        self.particle_size = particle_size
        self.down_particle_size = down_particle_size
        self.particle_filter = []
        self.beats_per_bar = beats_per_bar
        self.threshold = threshold
        self.fps = fps
        self.observation_lambda = observation_lambda
        self.plot = plot

        beats_per_bar = np.array(beats_per_bar, ndmin=1)
        min_bpm = np.array(min_bpm, ndmin=1)
        max_bpm = np.array(max_bpm, ndmin=1)
        num_tempi = np.array(num_tempi, ndmin=1)
        lambda_b = np.array(lambda_b, ndmin=1)
        # convert timing information to construct a beat state space
        if state_flag == 0:
            min_interval = 60. * fps / max_bpm
            max_interval = 60. * fps / min_bpm
        elif state_flag == 1:
            min_interval = 4 * 60. * fps / max_bpm
            max_interval = 4 * 60. * fps / min_bpm
        # model the different bar lengths

        self.meters = []

        st = BarStateSpace(1, min_interval[0], max_interval[0],    # beat tracking state space
                           num_tempi[0])
        if beats_per_bar:
            st2 = BarStateSpace(1, min(beats_per_bar),max(beats_per_bar), max(beats_per_bar)-min(beats_per_bar)+1)    #    downbeat tracking state space
        else:
            st2 = BarStateSpace(1, 2, 6, 5)        # downbeat tracking state space

        tm = BarTransitionModel(st, lambda_b[0])
        tm1 = list(TransitionModel.make_dense(tm.states, tm.pointers, tm.probabilities))
        om = BDObservationModel(st, observation_lambda)
        st.last_states = list(np.concatenate(st.last_states).flat)
        self.meters.append(meter(st, tm, tm1, om))
        self.st2 = st2
        self.om2 = BDObservationModel(st2, "B60")
        self.tm2=np.zeros((len(st2.first_states[0]),len(st2.first_states[0])))
        for i in range(len(st2.first_states[0])):
            for j in range(len(st2.first_states[0])):
                if i == j:
                    self.tm2[i, j] = 1-lambda2
                else:
                    self.tm2[i, j] = lambda2/(len(st2.first_states[0])-1)
            # self.particle_filter.append(HiddenMarkovModel(tm, om))
        pass
        # save variables

    def process(self, activations, **kwargs):

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
        # pylint: disable=arguments-differ
        import itertools as it
        # use only the activations > threshold (init offset to be added later)
        first = 0
        if self.threshold:
            idx = np.nonzero(activations >= self.threshold)[0]
            if idx.any():
                first = max(first, np.min(idx))
                last = min(len(activations), np.max(idx) + 1)
            else:
                last = first
            activations = activations[first:last]
        # return no beats if no activations given / remain after thresholding
        if not activations.any():
            return np.empty((0, 2))

        T = 1 / self.fps

        for counter, meter in enumerate(self.meters):
            counter = 0
            if self.plot:
                fig = plt.figure(figsize=(1800 / 96, 900 / 96), dpi=96)
                subplot1 = fig.add_subplot(211)
                subplot2 = fig.add_subplot(212)
                # line1 = subplot1.scatter(meter.st.state_positions,np.max(meter.st.state_intervals)-meter.st.state_intervals, marker='o', color='grey', alpha=0.08)
                # line2 = subplot2.scatter(self.st2.state_positions, np.max(self.st2.state_intervals) - self.st2.state_intervals, marker='o', color='grey', alpha=0.08)

            path = np.zeros((1, 2), dtype=float)
            position = []
            # particles = np.sort(np.random.randint(0, meter.st.num_states-1, self.particle_size))
            particles = np.sort(
                np.random.choice(np.arange(0, meter.st.num_states - 1), self.particle_size, replace=True))
            # down_particles = np.sort(
            #     np.random.choice(np.arange(0, self.beats_per_bar), self.down_particle_size, replace=True))
            down_particles = np.sort(
                np.random.choice(np.arange(0, self.st2.num_states - 1), self.down_particle_size, replace=True))
            beat = np.squeeze(meter.st.first_states)
            activations = activations[200:]
            both_activations = activations.copy()
            activations = np.max(activations, axis=1)
            activations[activations < 0.4] = 0.03

            for i in range(len(activations)):  # loop through all frames to infer beats/downbeats
                counter += 1
                if self.plot: # activate this when you want to plot the performance
                    position = np.median(meter.st.state_positions[particles])
                gathering = int(np.median(particles))
                if ((gathering - beat[meter.st.state_intervals[beat] == meter.st.state_intervals[gathering]]) < (
                int(.07 / T)) + 1).any() and (4 + counter * T) - path[-1][0] > .4 * T * meter.st.state_intervals[
                    gathering]:
                    last1 = down_particles[np.in1d(down_particles, self.st2.last_states)]
                    state1 = down_particles[~np.in1d(down_particles, self.st2.last_states)] + 1
                    for j in range(len(last1)):
                        arg1 = np.argwhere(self.st2.last_states[0] == last1[j])[0][0]
                        nn = np.random.choice(self.st2.first_states[0], 1, p=(np.squeeze(self.tm2[arg1])))
                        state1 = np.append(state1, nn)
                    down_particles = state1
                    obs2 = densities2(both_activations[i], self.om2, self.st2)
                    down_particles = resample(down_particles, obs2[down_particles])  # meter.st
                    m = np.bincount(down_particles)
                    down_max = np.argmax(m)
                    if down_max in self.st2.first_states[0]:
                        path = np.append(path, [[4 + counter * T, 1]], axis=0)
                    else:
                        path = np.append(path, [[4 + counter * T, 2]], axis=0)
                    if self.plot:
                        # m = m/np.max(m)
                        m1 = np.r_[True, down_particles[:-1] != down_particles[1:], True]
                        counts1 = np.diff(np.flatnonzero(m1))
                        unq1 = down_particles[m1[:-1]]
                        part1 = np.c_[unq1, counts1]
                        # fig = plt.figure(figsize=(400 / 96, 400 / 96), dpi=96)
                        subplot2.clear()
                        subplot2.scatter(self.st2.state_positions, np.max(self.st2.state_intervals) - self.st2.state_intervals,
                                    marker='o',
                                    color='grey', alpha=0.2, s=50)

                        subplot2.scatter(self.st2.state_positions[self.om2.pointers == 2],
                                    np.max(self.st2.state_intervals) - self.st2.state_intervals[self.om2.pointers == 2],
                                    marker='o',
                                    color='green', s=50, alpha=both_activations[i][1])
                        subplot2.scatter(self.st2.state_positions[self.om2.pointers != 2],
                                    np.max(self.st2.state_intervals) - self.st2.state_intervals[self.om2.pointers != 2],
                                    marker='o',
                                    color='orange', s=50, alpha=both_activations[i][0])
                        # plt.xlabel("ϕ", size=20)
                        # h = plt.ylabel("ϕ'  ", size=20)
                        # h.set_rotation(0)
                        # plt.xticks([])
                        # plt.yticks([])
                        # plt.scatter(meter.st.state_positions[beat], np.max(meter.st.state_intervals) - meter.st.state_intervals[beat], marker='o',
                        #             color='yellow', alpha= activations[i])
                        # for m in range (len(meter.st.state_positions)):
                        #     plt.scatter(meter.st.state_positions[m], 30 - meter.st.state_intervals[m], marker='o',
                        #             color='yellow', alpha=obs[m])
                        subplot2.scatter(self.st2.state_positions[part1[:, 0]],
                                    np.max(self.st2.state_intervals) - self.st2.state_intervals[part1[:, 0]],
                                    marker='o',
                                    s=part1[:, 1] * 4, color="red")
                        position2 = np.max(self.st2.state_positions[down_max])
                        subplot2.axvline(x=position2)


                # position = meter.st.state_positions[np.argmax(np.bincount(particles))]
                obs = densities(activations[i], meter.om, meter.st)
                if self.plot:
                    if counter % 1 == 0:  # counter > 0:  #
                        m = np.r_[True, particles[:-1] != particles[1:], True]
                        counts = np.diff(np.flatnonzero(m))
                        unq = particles[m[:-1]]
                        part = np.c_[unq, counts]
                        subplot1.scatter(meter.st.state_positions, np.max(meter.st.state_intervals) - meter.st.state_intervals,
                                    marker='o',
                                    color='grey', alpha=0.2)
                        if both_activations[i][0]>both_activations[i][1]:
                            subplot1.scatter(meter.st.state_positions[meter.om.pointers == 2],
                                        np.max(meter.st.state_intervals) - meter.st.state_intervals[meter.om.pointers == 2],
                                        marker='o',
                                        color='yellow', alpha=activations[i])
                        else:
                            subplot1.scatter(meter.st.state_positions[meter.om.pointers == 2],
                                        np.max(meter.st.state_intervals) - meter.st.state_intervals[meter.om.pointers == 2],
                                        marker='o',
                                        color='green', alpha=activations[i])
                        plt.xlabel("ϕ", size=20)
                        h = plt.ylabel("ϕ'  ", size=20)
                        h.set_rotation(0)
                        plt.xticks([])
                        plt.yticks([])
                        subplot1.scatter(meter.st.state_positions[part[:, 0]],
                                    np.max(meter.st.state_intervals) - meter.st.state_intervals[part[:, 0]],
                                    marker='o',
                                    s=part[:, 1] * 2, color="red")
                        subplot1.axvline(x=position)
                        # plt.show()
                        # plt.savefig(f"C:/research\downbeat/video5/{counter}.png")
                        plt.pause(0.002)
                        subplot1.clear()

                if activations[i] > 0.1:    # resampling is done only when there is a meaningful activation
                    particles = resample(particles, obs[particles], ) #meter.st
                last = particles[np.in1d(particles, meter.st.last_states)]
                state = particles[~np.in1d(particles, meter.st.last_states)] + 1
                for j in range(len(last)):
                    args = np.argwhere(meter.tm[1] == last[j])
                    probs = meter.tm[2][args]
                    nn = np.random.choice(np.squeeze(meter.tm[0][args]), 1, p=(np.squeeze(probs)))
                    state = np.append(state, nn)
                particles = state
                # if i %400==0 :
                #     particles = np.sort(np.random.randint(0, meter.st.num_states - 1, self.particle_size))

        return path[1:]



def resample(particles, weights, ): # state_space
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
    # median_interval = np.median(state_space.state_intervals[new_particles])
    # median_position = np.median(state_space.state_positions[new_particles])
    # if median_position<1/20:   #23/55<median_position<24/55:
    #     # if state_space.state_intervals[0]< median_interval*2<state_space.state_intervals[-1]:
    #     #     nn = np.random.choice(len(particles), 1)
    #     #     new_particles[nn] = state_space.first_states[0][np.where(state_space.state_intervals[state_space.first_states] ==  median_interval * 2)[0][0] ]         #investigates double temp  e.g. (28 -> 14) interval
    #     if state_space.state_intervals[0] <  median_interval * 0.5 < state_space.state_intervals[-1]:
    #         nn = np.random.choice(len(particles), 2)
    #         #new_particles[nn] = state_space.first_states[i][state_space.state_intervals[state_space.first_states[i]] == int(state_space.state_intervals[median] * 0.5)]   #investigates double tempo
    #         new_particles[nn] = state_space.first_states[0][np.where(state_space.state_intervals[state_space.first_states] == round(median_interval * .5))[0][0]]  # investigates double tempo
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

