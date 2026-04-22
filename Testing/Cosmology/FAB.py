from diffeqpy import de

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()

from diffeqpy import de
import numpy as np

f = lambda u,p,t: -u
u0 = 1.0
tspan = (0., 1.)
prob = de.ODEProblem(f, u0, tspan)
sol = de.solve(prob)
print(sol.t, sol.u)