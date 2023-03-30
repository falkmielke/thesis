#!/usr/bin/env python3

### Libraries
import numpy as NP

import matplotlib as MP
import matplotlib.pyplot as PLT


### Helper Functions
two_pi_j = 1j*2*NP.pi
MakeComplex = lambda twocolumnvector: NP.add(twocolumnvector[:,0], 1j*twocolumnvector[:,1])


### Fourier Series Decomposition
# take a signal from signal space to frequency space
# i.e. give Fourier coefficients c_n for a signal
SingleCoefficient = lambda t, T, n: NP.exp(-two_pi_j*n*t/T) / (2 if n == 0 else 1)
#   Note: the zero'th coefficient must be multiplied by two (complex conjugate doubling)

FourierSeriesDecomposition = lambda time, period, signal, order: \
                                        ( 2/period ) \
                                        * NP.array([ NP.sum(signal*SingleCoefficient(time, period, n))/len(time)  \
                                               for n in range(order+1) \
                                             ])



### Fourier Series Synthesis
# reconstruct a signal from its frequency space representation
# i.e. take coefficients and return the signal
FourierSummand = lambda t, T, cn, n: cn*NP.exp(two_pi_j*n*t/T)
FourierFunctionT = lambda t, T, coeffs: NP.sum(NP.array([ FourierSummand(t, T, cn, n) \
                                                           for n,cn in enumerate(coeffs) \
                                                          ]))

FourierReconstruction = lambda time, period, coeffs: NP.real(period \
                                                    * NP.array([FourierFunctionT(t, period, coeffs) for t in time]) \
                                                            )



### Example Data
coefficients_array = NP.stack([[0.06,  0.05, -0.12, -0.03,  0.01, -0.00], [0.00,  0.21,  0.06, -0.04, -0.01, -0.01]], axis = 1)
coefficients_C = MakeComplex(coefficients_array)
# print (coefficients_C)

# FSD parameters
order  = coefficients_array.shape[0]-1
period = 1.
time   = NP.linspace(0., period, 100, endpoint = False)

### Example Procedure
# reconstruct the signal
signal = FourierReconstruction(time, period, coefficients_C)


# retrieve coefficients
recovered_coefficients = FourierSeriesDecomposition(time, period, signal, order)
print(NP.sum(NP.abs(recovered_coefficients - coefficients_C)))

# ... and reconstruct the signal again
recovered_signal = FourierReconstruction(time, period, recovered_coefficients)
print (NP.sum(NP.abs(NP.subtract(signal, recovered_signal))))


## inspection plot
PLT.plot(time, signal)
PLT.plot(time, recovered_signal, 'g:')
PLT.show()
