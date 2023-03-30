### Helper Functions
two_pi_i <- 2*pi*1i

NameAlong0 <- function (arr) setNames(arr, seq.int(0, length(arr)-1))
MakeComplex <- function(twocolumnvector) NameAlong0(twocolumnvector[, 1] + 1i*twocolumnvector[,2])


### Fourier Series Decomposition
# take a signal from signal space to frequency space
# i.e. give Fourier coefficients c_n for a signal
SingleCoefficient <- function(n, t, T) exp(-two_pi_i*n*t/T) / ifelse(n == 0, 2., 1.)
#   Note: the zero'th coefficient must be multiplied by two (complex conjugate doubling)

DecompositionFunction <- function (n, time, period, signal)   sum(
                               signal * SingleCoefficient(n, time, period)
                               ) / length(time)

FourierSeriesDecomposition <- function (time, period, signal, order) (
                            ( 2/period ) * NameAlong0(
                               sapply(seq(0, order), DecompositionFunction, time, period, signal)
                               ))


### Fourier Series Synthesis
# reconstruct a signal from its frequency space representation
# i.e. take coefficients and return the signal
FourierSummand <- function (nplus1, coeffs, t, T) coeffs[nplus1]*exp(two_pi_i*(nplus1-1)*t/T)

FourierFunctionT <- function (t, T, coeffs) sum(
                            unlist(sapply(seq_along(coeffs), FourierSummand, coeffs, t, T))
                            )

FourierReconstruction <- function (time, period, coeffs) Re(
                                                    period * sapply(time, FourierFunctionT, period, coeffs)
                                                            )



### Example Data
# test coefficients
coefficients_array = cbind(c(0.06,  0.05, -0.12, -0.03,  0.01, -0.00), c(0.00,  0.21,  0.06, -0.04, -0.01, -0.01))
rownames(coefficients_array) = seq.int(0, 5)
colnames(coefficients_array) = c('re', 'im')

coefficients_C <- MakeComplex(coefficients_array)

# time
order  <- dim(coefficients_array)[1]-1
period <- 1.
time   <- seq(from = 0., to = 0.99, by = 1/100)


### Example Procedure
# reconstruct the signal
signal = FourierReconstruction(time, period, coefficients_C)
# print (signal)

# retrieve coefficients
recovered_coefficients = FourierSeriesDecomposition(time, period, signal, order)
print (coefficients_C)
print (recovered_coefficients)

# ... and reconstruct the signal again
recovered_signal = FourierReconstruction(time, period, recovered_coefficients)
# print (recovered_signal)


## inspection plot
# # test plot
# pdf("test.pdf")
# plot (time, signal)
# lines(time, recovered_signal, col="green")
# dev.off()
