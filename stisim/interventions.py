from . import utils as ssu
from . import modules as ssm


class Intervention(ssm.Module):
    pass


class Interventions(ssu.ndict):
    def __init__(self, *args, type=Intervention, **kwargs):
        return super().__init__(self, *args, type=type, **kwargs)
