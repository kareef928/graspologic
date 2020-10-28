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

n_components = 2
n_components_2 = 4  # the number of embedding dimensions for ASE
P = np.array([[0.9, 0.6], [0.6, 0.9]])
csize = [150] * 2
A1 = sbm(csize, P)
A2 = sbm(csize, P)

np.random.seed(8888)
lpt_class = LatentPositionTest(n_bootstraps=150, n_components=n_components)
lpt_class.fit(A1, A2)
print(lpt_class.p_value_)

np.random.seed(8888)
p_val, _, _ = lpt_function(A1, A2, n_bootstraps=150, n_components=n_components)
print(p_val)

csize2 = [150] * 4
csize3 = [250] * 4
P2 = np.array(
    [[0.9, 0.11, 0.13, 0.2], [0, 0.7, 0.1, 0.1], [0, 0, 0.8, 0.1], [0, 0, 0, 0.85]]
)
P2 = symmetrize(P2)
A3 = sbm(csize2, P2)
A4 = sbm(csize3, P2)

np.random.seed(888)
lpt_class = LatentDistributionTest(
    n_bootstraps=500, n_components=n_components_2, size_correction=True
)
lpt_class.fit(A3, A4)
print(lpt_class.p_value_)

np.random.seed(888)
p_val, _, _ = ldt_function(
    A3, A4, n_bootstraps=500, n_components=n_components_2, size_correction=True
)
print(p_val)

