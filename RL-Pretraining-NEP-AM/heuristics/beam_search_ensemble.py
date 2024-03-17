from heuristics.beam_search import BeamSearch


class EnsembleBeamSearch:
    def __init__(self, models, width):
        """
        :param models: A List of BeamSearchModel
        :param width: Number of Children.
        """
        super(EnsembleBeamSearch, self).__init__()
        self.beam_search_list = [BeamSearch(models[i], width) for i in range(len(models))]

    def ensemble_beam_search(self, env):
        res = []
        for beam_search in self.beam_search_list:
            res.extend(beam_search.beam_search(env))
        return res


if __name__ == '__main__':
    pass
