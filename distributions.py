class Distribution:

    ...

class Discrete(Distribution):

    ...

class Categorical(Discrete):

    def __init__(self, symbols, parameters):
        """Initialise categorical distribution with given parameters.

        Parameters must be of length symbols minus one.
        """
        self.set_parameters(symbols, parameters)

    def set_parameters(self, symbols, parameters):
        assert len(parameters) == len(symbols) - 1
        assert sum(parameters) <= 1
        self.symbols = symbols
        self.distribution = {s:p for s, p in zip(symbols, parameters + [1 - sum(parameters)])}


    def probability(self, symbol):
        return self.distribution[symbol]

class Bernouilli(Categorical):

    def __init__(self, parameter, symbols=[True, False]):
        super().__init__(symbols, [parameter])

class Conditional:

    def __init__(self, symbols, distributions):
        self.distributions = {s:d for s, d, in zip(symbols, distributions)}

    def distribution(self, symbol):
        return self.distributions[symbol]

    def conditional_probability(self, conditional, symbol):
        distribution = self.distribution(conditional)
        return distribution.probability(symbol)

