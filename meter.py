import numpy as np, argparse, notefile
from tools import memoized
from itertools import product as cartesian_product
from functools import reduce
from distributions import Categorical, Bernouilli, Conditional
from matplotlib import pyplot as plt

## Custom probability distributions

class FirstTactus(Categorical):

    def __init__(self, min_tactus, max_tactus):
        span = max_tactus - min_tactus
        assert(span > 0 and span <= 23 - 9)
        symbols = [t for t in range(min_tactus, max_tactus)]
        params = [.10, .20, .30, .23, .13, .03, .006, .002, .001, .0006, .0002, .0001, .00005, .00005]
        mode = 2
        left = max(mode - (span // 2), 0)
        right = left + span
        params = params[left:right]
        params = [p/sum(params) for p in params]
        super().__init__(symbols, params[:-1])

class Regularity(Categorical):

    def __init__(self, span=3):
        assert span < 4 and span > 0
        self.span = span
        symbols = np.arange(-span, span+1)
        params = [.02, .08, .24, .32, .24, .08, .02]
        params = params[3-span:4 + span]
        params = [p/sum(params) for p in params]
        super().__init__(symbols, params[:-1])

class BeatLocation(Regularity):

    def __init__(self, tactus_interval, subdivision, beat, *args, **kwargs):
        super().__init__(*args, **kwargs)
        beat_length = tactus_interval // subdivision
        idealised = beat_length * beat
        symbols = self.symbols
        parameters = np.asarray([self.probability(s) for s in symbols])

        symbols += idealised
        maximum  = idealised + self.span
        minimum = idealised - self.span
        if maximum > tactus_interval:
            overflow = maximum - tactus_interval 
            symbols = symbols[:-overflow]
            parameters = parameters[:-overflow]
        if minimum < 0:
            symbols = symbols[-minimum:]
            parameters = parameters[-minimum:]
        parameters /= sum(parameters)
        self.set_parameters(symbols, list(parameters)[:-1])

class TactusInterval(Conditional):
    """"Conditional distribution representing the probability of 
    a tactus interval given a previous tactus interval.
    """

    def __init__(self, min_tactus, max_tactus):
        intervals = list(range(min_tactus, max_tactus))
        conditionals = intervals
        distributions = []
        for first in intervals:
            params = []
            for second in intervals:
                params.append(np.exp(-(.5 * abs(first - second)) ** 2))
            normalised = [p/sum(params) for p in params]
            distribution = Categorical(intervals, normalised[:-1])
            distributions.append(distribution)
        super().__init__(conditionals, distributions)

# The meter model

class MeterModel:

    def __init__(self, min_tactus=9, max_tactus=23, beat_span=3, subdivisions=[[2,3], [2,3]]):
        """Initialize a generative meter model.

        Keyword arguments:
        min_tactus -- integer specifying minimum tactus interval (default 9)
        min_tactus -- maximum tactus interval (default 23)
        beat_span -- number of pips before and after ideal beat location to consider (default 3)
        subdivisions -- list of lists where each list specifies the subdivision possibilities for 
            the corresponding level (default [[2,3],[2,3]])
        """
        # Generative parameters
        self.beat_span = beat_span
        self.min_tactus = min_tactus
        self.max_tactus = max_tactus
        # Probability distributions
        self.tactus_interval = TactusInterval(min_tactus, max_tactus)
        self.subdivisions = subdivisions
        self.metrical_levels = len(subdivisions) 
        self.levels=[Bernouilli(.78, symbols=self.subdivisions[0]), Bernouilli(.76, symbols=self.subdivisions[1])]
        self.phase=[
            Conditional(
                self.subdivisions[1], 
                [Bernouilli(.65, symbols=[0, 1]), Categorical([0, 1, 2], [.33, .67 * .004])],
            ),
        ]
        self.note=Conditional(
            [0, 1, 2, 3],
            [Bernouilli(.001), Bernouilli(.38), Bernouilli(.74), Bernouilli(.95)],
        )
        self.first_tactus=FirstTactus(min_tactus, max_tactus)
        self.note_first_tactus=Bernouilli(.6)
        self.another_beat=Bernouilli(.95)

    @property
    def meters(self):
        results = []
        for (subdivision, grouping) in cartesian_product(*[self.subdivisions[level] for level in range(self.metrical_levels)]):
            for phase in range(grouping):
                results.append(((subdivision, grouping), phase))
        return results

    @property
    def tactus_intervals(self):
        return np.arange(self.min_tactus, self.max_tactus)

    def tactus_spans(self, rhythm_length, begin=1):
        spans = []
        last_pip = rhythm_length - 1
        for end in range(begin, rhythm_length + self.max_tactus):
            for interval in self.tactus_intervals:
                begin = end - interval
                if begin > last_pip:
                    continue
                # First tactus beat?
                first_beat = begin <= 0
                # Last tactus beat?
                last_beat = end > last_pip
                spans.append((begin, end, first_beat, last_beat))
        return spans

    @memoized
    def beat_location(self, tactus_interval, subdivision, beat):
        # Make memoised
        return BeatLocation(tactus_interval, subdivision, beat, span=self.beat_span)

    def beat_probability(self, salience, onset):
        return self.note.conditional_probability(salience, onset)

    def meter_probability(self, levels, phases):
        assert len(levels) - 1 == len(phases) 
        meter = [d.probability(l) for d, l in zip(self.levels, levels)]
        phase = [cd.conditional_probability(m, phase) for cd, m, phase in zip(self.phase, levels[1:], phases)]
        return reduce(float.__mul__, meter + phase)
                
def argmax_index(parameters, f):
    return max(range(len(parameters)), key=lambda i: f(parameters[i]))


# Not that fast because entire rhythm is passed and hashed?
@memoized
def interval_probability(model, excerpt, beats):
    probabilities = [
        (
            np.log(model.beat_probability(1, onset)) if position + 1 in beats else 
            np.log(model.beat_probability(0, onset))
        ) for position, onset in enumerate(excerpt)
    ]
    return sum(probabilities)

def generate_tactus_table(model, rhythm):

    print('generating tactus analyses.')
    probabilities, analyses = {}, {}

    for begin, end, _, _ in model.tactus_spans(len(rhythm)):
        overflow = end - len(rhythm)
        padding = (- begin - 1 if begin < 0 else 0)
        left = (0 if begin < 0 else begin + 1)
        if overflow < 0: 
            overflow = 0
        exerpt = (False, ) * padding + rhythm[left:end] + overflow * (False, )
        interval = end - begin
        for subdivision in model.subdivisions[0]:
            #print('[%d to %d)\tsubdivided by %d.' % (end-interval, end, subdivision))
            beat_dists = [model.beat_location(interval, subdivision, b) for b in range(1, subdivision)]
            # All possible sequences of beat locations
            beat_locations = list(cartesian_product(*(d.symbols for d in beat_dists)))
            # PROBS
            beats_prob_f = lambda beats: sum([np.log(d.probability(b)) for b, d in zip(beats, beat_dists)])
            probability_f = lambda beats: beats_prob_f(beats) + interval_probability(model, exerpt, beats)
            i =  argmax_index(beat_locations, probability_f)
            analyses[subdivision, begin, end] = beat_locations[i]
            probabilities[subdivision, begin, end] = probability_f(beat_locations[i])
            #print(probabilities[subdivision, begin, end])

    return probabilities, analyses

def generate_meter_table(model, rhythm, lower_level_probabilities):

    print('generating table.')

    probabilities, analyses = {}, {}
    onset = lambda pip: (rhythm[pip] if pip >= 0 and pip < len(rhythm) else False)
    meters = model.meters 
    # Mass?
    for begin, end, first_beat, last_beat in model.tactus_spans(len(rhythm)):
        interval = end - begin
        for (subdivision, grouping), phase in meters:
            #print('[%d to %d)%s%s\t%s' % (begin, end, ('^' if first_beat else ''), ('$' if last_beat else ''), (phase, subdivision, grouping)))
            # <phase> is interpreted to correspond to the phase at the <end>
            # <previous_phase> corresponds to the beat at <begin>
            previous_phase = (phase - 1) % grouping
            # is the beat at <end> strong?
            strong = phase % grouping == 0

            def probability_f(previous_interval):
                """The probability of the current tactus interval preceded by
                a hypothetical previous interval.
                """
                return (
                    # Probability of best analysis of current interval (independent of grouping and phase)
                    lower_level_probabilities[subdivision, begin, end] +
                    # Probability of of onset at current tactus beat
                    np.log(model.beat_probability(3 if strong else 2, onset(end))) +
                    (
                        # Prior probability of the meter and phase
                        np.log(model.meter_probability([subdivision, grouping], [phase])) +
                        # Probability of a note on the first tactus beat
                        np.log(model.note_first_tactus.probability(onset(begin))) +
                        # Probability of first tactus interval duration
                        np.log(model.first_tactus.probability(interval))
                        # Theoretically, below would make sense
                        # + model.beat_probability(3 if strong else 2, onset(end)) +
                        if first_beat else 
                        probabilities[subdivision, grouping, previous_phase, begin-previous_interval, begin] +
                        np.log(model.tactus_interval.conditional_probability(previous_interval, interval)) + 
                        np.log(model.another_beat.probability(last_beat))
                    )
                )

            i = argmax_index(model.tactus_intervals, probability_f)
            # Given the current interval, store the best previous interval and the probability of that interval followed by
            # the present one.
            analyses[subdivision, grouping, phase, begin, end] = model.tactus_intervals[i]
            probabilities[subdivision, grouping, phase, begin, end] = probability_f(model.tactus_intervals[i])

    return probabilities, analyses

def best_analysis(model, rhythm, probabilities):
    parameters = []
    meters = model.meters

    for begin, end, _, _ in model.tactus_spans(len(rhythm), begin=len(rhythm)):
        for (subdivision, grouping), phase in model.meters:
            parameters.append((subdivision, grouping, phase, begin, end))

    best = max(parameters, key=lambda params: probabilities[params])
    return best, probabilities[best]
    

def trace_back(analyses, tactus_analyses, subdivision, grouping, phase, begin, end,
        subtactus_beat_locations=[], tactus_beats=[]):
    if end > 0:
        tactus = end
        subtactus_beats = tactus_analyses[subdivision, begin, end]
        previous_interval = analyses[subdivision, grouping, phase, begin, end]
        return trace_back(
            analyses, tactus_analyses,
            subdivision, grouping, (phase - 1) % grouping, begin-previous_interval, begin,
            tactus_beats=[tactus] + tactus_beats,
            subtactus_beat_locations=[subtactus_beats] + subtactus_beat_locations
        )
    else:
        return tactus_beats, subtactus_beat_locations, (end, phase, subdivision, grouping)

def almost_equal(p1, p2, eta=1e-10):
    return abs(p1 - p2) < eta

def run():
    np.seterr(divide='warn')

    parser = argparse.ArgumentParser()
    parser.add_argument('--note_file')
    args = parser.parse_args()

    model = MeterModel()

    two_two = .78 * .76
    assert model.meter_probability([2, 2], [0]) == two_two * .65
    assert model.meter_probability([2, 2], [1]) == two_two * .35
    assert almost_equal(model.meter_probability([2, 3], [2]), .78 * .24 * (.67 * .996))
    assert almost_equal(model.meter_probability([2, 3], [0]), .78 * .24 * (.33))

    #tactus_distribution = model.tactus_interval
    #for tactus_interval in [9, 15, 20, 22]:
    #    beat_location = model.beat_location(tactus_interval, 3, 1)
    #    plt.figure()
    #    plt.title('tactus interval: %s' % tactus_interval)
    #    #plt.plot(model.tactus_intervals, list(map(lambda i: tactus_distribution.conditional_probability(tactus_interval, i), model.tactus_intervals)))
    #    x = beat_location.symbols
    #    plt.plot(x, list(map(lambda i: beat_location.probability(i), x)))
    #plt.show()

    model = MeterModel()
    #rhythm = [True, False, True]
    if args.note_file is None:
        return 

    with open(args.note_file) as f:
        lines = [l for l in f]
        onsets = notefile.onsets(lines)
        rhythm = tuple(notefile.pips(onsets))
    tactus_probabilities, tactus_analyses = generate_tactus_table(model, rhythm)
    probabilities, analyses = generate_meter_table(model, rhythm, tactus_probabilities)
    meter, ll = best_analysis(model, rhythm, probabilities)
    tactus_beats, subtactus_beat_locations, (first_beat, phase, subdivision, grouping) = trace_back(analyses, tactus_analyses, *meter)
    tb = [first_beat] + tactus_beats
    print(tb)
    print([b - a for a, b in zip(tb[:-1], tb[1:])])
    print(subtactus_beat_locations)
    print('phase %s, subdivision %s, grouping %s, likelihood %s' % (phase, subdivision, grouping, ll))

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def print_analysis(rhythm, meter):
    line_width = 80
    for chunk in chunks(enumerate(rhythm), line_width):
        (phase, onset), *rest = chunk
        l3 = '' 
        l2 = ''
        l1 = ''
        r = ''
        print('%s\n%s\n%s\n\n%s\n\n' % (l3, l2, l1, r))

if __name__ == '__main__':
    run()
