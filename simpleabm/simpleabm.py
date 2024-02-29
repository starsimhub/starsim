'''
Simple agent-based network model in Python

NOTE: How to view the animation depends on what IDE you're using.
Specifically:
- It will work without change from the command line, Spyder, or VS Code.
- For PyCharm, ensure you disable "SciView" before running.
- For Jupyter, set save_movie = True and view the saved movie
  (you might need to run "pip install ffmpeg-python" in a terminal first)
'''

import numpy as np
import sciris as sc
import pylab as pl

sc.options(dpi=200)

__all__ = ['default_pars', 'Person', 'Sim']

# Set default parameters
default_pars = sc.objdict(
    beta = 0.3, # Infection rate per contact per unit time
    gamma = 0.5, # Recovery rate
    n_contacts = 10, # Number of people each person is connected to
    distance = 0.1, # The distance over which people form contacts
    I0 = 1, # Number of people initially infected
    N = 1000, # Total population size
    maxtime = 20, # How long to simulate for
    npts = 100, # Number of time points during the simulation
    seed = 4, # Random seed to use
    colors = sc.dictobj(S='darkgreen', I='gold', R='skyblue'),
    save_movie = False, # Whether to save the movie (slow)
)


# Define each person
class Person(sc.dictobj):

    def __init__(self, pars):
        self.pars = pars
        self.S = True # People start off susceptible
        self.I = False
        self.R = False
        self.x = np.random.rand()
        self.y = np.random.rand()

    def infect(self):
        self.S = False
        self.I = True

    def recover(self):
        self.I = False
        self.R = True

    def check_infection(self, other):
        pars = self.pars
        if self.S: # A person must be susceptible to be infected
            if other.I: # The other person must be infectious
                if np.random.rand() < pars.beta*pars.dt: # Infection is probabilistic
                    self.infect()

    def check_recovery(self):
        pars = self.pars
        if self.I: # A person must be infected to recover
            if np.random.rand() < pars.gamma*pars.dt: # Recovery is also probabilistic
                self.recover()


# Define the simulation
class Sim(sc.dictobj):

    def __init__(self, **kwargs):
        
        # Initialize parameters
        pars = sc.mergedicts(default_pars, kwargs) # Parameters to use
        pars.dt = pars.maxtime/pars.npts # Timestep length
        self.T = np.arange(pars.npts)
        self.time = self.T*pars.dt
        self.pars = pars
        self.initialize()
    
    def initialize(self):
        """ Initialize everything (sim can be re-initialized as well) """
        pars = self.pars
        
        # Initilaize people and the network
        np.random.seed(pars.seed)
        self.people = [Person(pars) for i in range(pars.N)] # Create all the people
        for person in self.people[0:pars.I0]: # Make the first I0 people infectious
            person.infect() # Set the initial conditions
        self.make_network()
        
        # Initial conditions
        self.S = np.zeros(pars.npts)
        self.I = np.zeros(pars.npts)
        self.R = np.zeros(pars.npts)
        self.S_full = []
        self.I_full = []
        self.R_full = []
        
    def get_xy(self):
        """ Get the location of each agent """
        x = np.array([p.x for p in self.people])
        y = np.array([p.y for p in self.people])
        return x,y
        
    def make_network(self):
        pars = self.pars
        x,y = self.get_xy()
        dist = np.zeros((pars.N, pars.N))
        for i in range(pars.N):
            dist[i,:] = 1 + ((x - x[i])**2 + (y - y[i])**2)**0.5/self.pars.distance
            dist[i,i] = np.inf
            
        rnds = np.random.rand(pars.N, pars.N)
        ratios = dist/rnds
        order = np.argsort(ratios, axis=None)
        inds = order[0:int(pars.N*pars.n_contacts/2)]
        contacts = np.unravel_index(inds, ratios.shape)
        self.contacts = np.vstack(contacts).T

    def check_infections(self): # Check which infectious occur
        for p1,p2 in self.contacts:
            person1 = self.people[p1]
            person2 = self.people[p2]
            person1.check_infection(person2)
            person2.check_infection(person1)

    def check_recoveries(self): # Check which recoveries occur
        for person in self.people:
            person.check_recovery()
    
    def count(self, t):
        this_S = []
        this_I = []
        this_R = []
        for i,person in enumerate(self.people):
            if person.S: this_S.append(i)
            if person.I: this_I.append(i)
            if person.R: this_R.append(i)
        
        self.S[t] += len(this_S)
        self.I[t] += len(this_I)
        self.R[t] += len(this_R)
        
        self.S_full.append(this_S)
        self.I_full.append(this_I)
        self.R_full.append(this_R)
            
    def run(self):
        for t in self.T:
            self.check_infections() # Check which infectious occur
            self.check_recoveries() # Check which recoveries occur
            self.count(t) # Store results

    def plot(self):
        pl.figure()
        cols = self.pars.colors
        pl.plot(self.time, self.S, label='Susceptible', c=cols.S)
        pl.plot(self.time, self.I, label='Infectious', c=cols.I)
        pl.plot(self.time, self.R, label='Recovered', c=cols.R)
        pl.legend()
        pl.xlabel('Time')
        pl.ylabel('Number of people')
        pl.ylim(bottom=0)
        pl.xlim(left=0)
        pl.show()
        
    def animate(self, pause=0.01, save=False):
        anim = sc.animation()
        fig,ax = pl.subplots()
        x,y = self.get_xy()
        for p in self.contacts:
            p0 = p[0]
            p1 = p[1]
            pl.plot([x[p0], x[p1]], [y[p0], y[p1]], lw=0.5, alpha=0.1, c='k')
            
        handles = []
        for t in self.T[:-1]:
            if pl.fignum_exists(fig.number):
                for h in handles:
                    h.remove()
                handles = []
                counts = sc.dictobj()
                inds = sc.dictobj()
                for key,fullkey in [('S', 'S_full'), ('I', 'I_full'), ('R', 'R_full')]:
                    inds[key] = self[fullkey][t]
                    counts[key] = len(inds[key])
                    this_x = x[inds[key]]
                    this_y = y[inds[key]]
                    h = ax.scatter(this_x, this_y, c=self.pars.colors[key])
                    handles.append(h)
                title = f't={t}, S={counts.S}, I={counts.I}, R={counts.R}'
                pl.title(title)
                pl.pause(pause)
                if save:
                    anim.addframe()
        
        if save:
            anim.save(f'network_{self.pars.distance}.mp4')
        
        
if __name__ == '__main__':
    
    save = False
    
    # Create and run the simulation
    sim = Sim()
    sim.run()
    sim.plot()
    sim.animate(save=save)