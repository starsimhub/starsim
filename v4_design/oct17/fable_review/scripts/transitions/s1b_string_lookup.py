"""Surprise: a string 'I' in ss.Transition is resolved to the caller's *variable* named I, not to the declared state named 'I'."""
import starsim as ss
S, I, R = ss.states(3)
sirA = ss.Disease(S >> I @ 0.3*I, I >> R @ 10, name='sirA')   # claims variables S, I, R

S3, I3, R3 = ss.states('S I R')     # a second set of states, with the same *names*
try:
    sirB = ss.Disease(ss.Transition(S3, I3, rate=0.3*I3, name='infection'),
                      ss.Transition('I', 'R', dur=10, name='recovery'), name='sirB')
    print('built OK:', sirB.graph)
except Exception as e:
    print('ERROR:', e)

# Same thing without any competing module: does 'I' resolve to I3 (the declared state called I)?
del I, R, S
S3, I3, R3 = ss.states('S I R')
sirC = ss.Disease(ss.Transition(S3, I3, rate=0.3*I3, name='infection'),
                  ss.Transition('I', 'R', dur=10, name='recovery'), name='sirC')
print('sirC states:', [ (n, id(s)==id(I3)) for n,s in sirC.graph.states.items()])
