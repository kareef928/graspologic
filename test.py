import numpy as np
import random
import matplotlib.pyplot as plt

from graspologic.inference import (
    lpt_function,
    LatentPositionTest,
    ldt_function,
    LatentDistributionTest,
)
from graspologic.embed import AdjacencySpectralEmbed
from graspologic.simulations import sbm, rdpg
from graspologic.utils import symmetrize
from graspologic.plot import heatmap, pairplot

n_components = 2  # the number of embedding dimensions for ASE
P = np.array([[0.9, 0.6], [0.6, 0.9]])
csize = [70] * 2
A1 = sbm(csize, P)
A2 = sbm(csize, P)

random.seed(100)
print(random.randint(300, 500))
random.seed(100)
print(random.randint(300, 500))

lpt_class = LatentPositionTest(n_bootstraps=150, n_components=n_components)
lpt_class.fit(A1, A2)
print(lpt_class.p_value_)

p_val, _, _ = lpt_function(A1, A2, n_bootstraps=150, n_components=n_components)
print(p_val)

