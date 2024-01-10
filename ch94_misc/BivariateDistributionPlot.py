#!/usr/bin/env python3

import LinearModelSimulation as LMS
import scipy.stats as STATS

sim = LMS.Simulation(slopes = [0.2] \
                        , intercept = 0.1 \
                        , x_range = [-0.5, 0.5] \
           )
# sim.FitModel()
# sim.PredictiveSampling()

fig, ax_dict = LMS.MakePlot()

LMS.PlotData(ax_dict, sim, color = '0.5', label = None, zorder = 0)

regression = sim.LinearRegression()
ax = ax_dict['ax']
reg_x = LMS.NP.array(sim.settings['x_range'])
reg_y = regression.intercept + regression.slope * reg_x
ax.plot(reg_x, reg_y, 'k-', label = f'regression: y = {regression.intercept:.2f} + {regression.slope:.2f} x')




fig.savefig(f"""./show/bivariate_distribution.png""", dpi = LMS.dpi)
LMS.PLT.close()
