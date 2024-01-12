#!/usr/bin/env python3

import LinearModelSimulation as LMS
import scipy.stats as STATS
import numpy as NP

sim = LMS.Simulation( slopes = [0.45] \
                    , intercept = 0.2 \
                    , x_range = [-0.25, 0.25] \
                    , n_observations = 2**8 \
                    , noise = 0.1
                   )
# sim.PredictiveSampling()

fig, ax_dict = LMS.MakePlot()

LMS.PlotData(ax_dict, sim, color = '0.5', label = None, zorder = 0)

regression = sim.LinearRegression()
ax = ax_dict['ax']
reg_x = NP.array(sim.settings['x_range'])
print (regression.intercept, regression.slope)
reg_y = regression.intercept + regression.slope * reg_x
ax.plot(reg_x, reg_y, 'k-', label = f'regression: y = {regression.intercept:.2f} + {regression.slope:.2f} x')

ax.set_xlabel("x"); ax.set_ylabel("y")
fig.savefig(f"""./show/bivariate_distribution.png""", dpi = LMS.dpi)
LMS.PLT.close()

# probabilistic variant
sim.FitModel()
fig, ax_dict = LMS.MakePlot()
ax = ax_dict['ax']
LMS.PlotData(ax_dict, sim, color = '0.5', label = None, zorder = 0)


for chain in sim.trace.posterior.chain:
    for draw in NP.random.choice(sim.trace.posterior.isel(chain = chain).draw.values \
                                 , 5):
        slope = sim.trace.posterior.isel(chain = chain, draw = draw).slopes.values.ravel()
        intercept = sim.trace.posterior.isel(chain = chain, draw = draw).intercept.values.ravel()
        print (intercept, slope)

        reg_y = intercept + slope * reg_x
        ax.plot(reg_x, reg_y, 'k-' \
                , label = f'regression: y = {regression.intercept:.2f} + {regression.slope:.2f} x' \
                , alpha = 0.1)

ax.set_xlabel("x"); ax.set_ylabel("y")
fig.savefig(f"""./show/bivariate_distribution_probabilistic.png""", dpi = LMS.dpi)
LMS.PLT.close()
