%% Helper Functions

two_pi_i = 1i*2*pi;

ReshapeComplex = @(arr) reshape(arr, 1, []);
MakeComplex = @(twocolumnvector) ReshapeComplex(twocolumnvector(:,1) + 1i*twocolumnvector(:,2));

%% Fourier Series Decomposition
% take a signal from signal space to frequency space
% i.e. give Fourier coefficients c_n for a signal
SingleCoefficient = @(n, t, T) exp(-two_pi_i*n*t/T) / (1. + double(n==0));
%   Note: the zero'th coefficient must be multiplied by two (complex conjugate doubling)

DecompositionFunction = @(n, time, period, signal)   sum( ...
                               signal .* SingleCoefficient(n, time, period) ...
                               ) / length(time) ...
                         ;

FourierSeriesDecomposition = @(time, period, signal, order) ReshapeComplex(...
                               ( 2/period ) ...
                               * arrayfun( @(n) DecompositionFunction(n, time, period, signal) ...
                                         , 0:order ) ...
                               );




%% Fourier Series Synthesis
% reconstruct a signal from its frequency space representation
% i.e. take coefficients and return the signal
FourierSummand = @(nplus1, coeffs, t, T) coeffs(nplus1)*exp(two_pi_i*(nplus1-1)*t/T);

FourierFunctionT = @(t, T, coeffs) sum( ...
                            arrayfun(@(nplus1) FourierSummand(nplus1, coeffs, t, T) ...
                                     , 1:numel(coeffs)) ...
                            );

FourierReconstruction = @(time, period, coeffs) real( period ...
                                                    * arrayfun(@(t) FourierFunctionT(t, period, coeffs) ...
                                                                , time) ...
                                                            );



%% Example Data
coefficients_array = [0.06,  0.05, -0.12, -0.03,  0.01, -0.00; 0.00,  0.21,  0.06, -0.04, -0.01, -0.01]';
coefficients_C = MakeComplex(coefficients_array);

% FSD parameters
order  = size(coefficients_array, 1)-1;
period = 1.;
time   = linspace(0., period-1/100, 100);


%% Example Procedure
% reconstruct the signal
signal = FourierReconstruction(time, period, coefficients_C);

% retrieve coefficients
recovered_coefficients = FourierSeriesDecomposition(time, period, signal, order);
disp (coefficients_C)
disp (recovered_coefficients)


% ... and reconstruct the signal again
recovered_signal = FourierReconstruction(time, period, recovered_coefficients);
% disp (recovered_signal)

%% inspection plot
% h = plot (time, signal);
% hold on;
% plot (time, recovered_signal, 'g--');
% waitfor(h);
