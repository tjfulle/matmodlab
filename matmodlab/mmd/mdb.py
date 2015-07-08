__all__ = ['mdb']

class ModelCaptured(Exception):
    pass

class MatModLabDB(object):
    '''Databse for holding meta information regarding models'''
    def __init__(self):
        self.models = []
        self.permutators = []
        self.optimizers = []

    def stash(self):
        self._models = [x for x in self.models]
        self.models = []
        self._permutators = [x for x in self.permutators]
        self.permutators = []
        self._optimizers = [x for x in self.optimizers]
        self.optimizers = []

    def pop_stash(self):
        self.permutators += self._permutators
        self.optimizers += self._optimizers
        self.models += self._models

    def add_model(self, job):
        self.models.append(job)

    def get_model(self, job=None):
        if job is None:
            try: return self.models[-1]
            except IndexError: return
        for item in self.models:
            if job and item.job == job:
                return item
        return

    def add_permutator(self, job):
        self.permutators.append(job)

    def get_permutator(self, job=None):
        if job is None:
            try: return self.permutators[-1]
            except IndexError: return
        for item in self.permutators:
            if job and item.job == job:
                return item
        return

    def add_optimizer(self, job):
        self.optimizers.append(job)

    def get_optimizer(self, job=None):
        if job is None:
            try: return self.optimizers[-1]
            except IndexError: return
        for item in self.optimizers:
            if job and item.job == job:
                return item
        return

mdb = MatModLabDB()
