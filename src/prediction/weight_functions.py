# -*- coding: utf-8 -*-

import numpy as np

class WeightFunction:

    def __init__(self):
        pass

    def predict(self, predictions, theta, times, text_lengths):
        weights = self.get_weights(predictions, times, text_lengths)
        prediction = np.average(predictions, weights=weights) > theta

        return prediction

    def get_weights(self, predictions, times, text_lenghts):
        raise Exception('Subclasses should implement this.')


class ExponentialNorm(WeightFunction):

    def __init__(self, lamb):
        self.lamb = lamb
        self.SECS_PER_MONTH = 60 * 60 * 24 * 30

    def get_weights(self, predictions, times, text_lengths):
        assert times.shape == predictions.shape
        assert times.shape == text_lengths.shape

        times = np.around((np.max(times) - times) / self.SECS_PER_MONTH)
        weights = self.exp_norm(times)

        return weights

    def exp_norm(self, xs):
        xs = np.exp(-self.lamb * xs)

        return xs / np.sum(xs)

    def __str__(self):
        return 'exp-norm {}'.format(self.lamb)


class MaximumWeight(WeightFunction):

    def __init__(self):
        pass

    def get_weights(self, predictions, times, text_lengths):
        assert times.shape == predictions.shape
        assert times.shape == text_lengths.shape

        weights = np.zeros(predictions.shape)
        weights[np.argmax(predictions)] = 1

        return weights

    def __str__(self):
        return 'maximum'


class MinimumWeight(WeightFunction):

    def __init__(self):
        pass

    def get_weights(self, predictions, times, text_lengths):
        assert times.shape == predictions.shape
        assert times.shape == text_lengths.shape

        weights = np.zeros(predictions.shape)
        weights[np.argmin(predictions)] = 1

        return weights

    def __str__(self):
        return 'minimum'


class TextLengthWeight(WeightFunction):

    def __init__():
        self.chars = 1000

    def get_weights(self, predictions, times, text_lengths):
        chunked = np.around(text_lengts / self.chars)
        return chunked / np.sum(chunked)

    def __str__(self):
        return 'text-length'


class MajorityVoteWeight(WeightFunction):

    def __init__(self):
        pass

    def predict(self, predictions, theta, times, text_lengths):
        above = np.count_nonzero(predictions > theta)
        below = np.count_nonzero(predictions <= theta)

        return above >= below

    def __str__(self):
        return 'majority-vote'


def weight_factory(weight_type):
    if weight_type == 'exp-norm' or weight_type == 'exponential-norm':
        return [ExponentialNorm(lamb) for lamb in np.linspace(0, 1, num=5)]
    elif weight_type == 'max' or weight_type == 'maximum':
        return [MaximumWeight()]
    elif weight_type == 'min' or weight_type == 'minimum':
        return [MinimumWeight()]
    elif weight_type == 'text':
        return [TextLengthWeight()]
    elif weight_type == 'majority-vote':
        return [MajorityVoteWeight()]
    else:
        raise Exception('Unknown weight {}'.format(weight_type))
