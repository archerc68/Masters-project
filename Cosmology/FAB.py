import numpy as np
from differint.differint import PCsolver

y_solved = PCsolver([1, 1], 1.5, lambda x, y : y - x - 1)
theoretical = np.linspace(0, 1, 100) + 1
same = np.allclose(y_solved, theoretical)

print(same)