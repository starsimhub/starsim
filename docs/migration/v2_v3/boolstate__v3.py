import starsim as ss

class MySIR:
    def __init__(self):
        super().__init__()
        self.define_states(
            ss.FloatArr('my_attr'),
            ss.BoolState('mystate', label='My new state'),
        )
        return 