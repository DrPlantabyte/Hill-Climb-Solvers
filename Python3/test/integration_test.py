import numpy
from random import random
from matplotlib import pyplot
import scipy.optimize
import hillclimb

actual_params = numpy.asarray([-0.05,0.2,1.5,-5.7])
initial_params = numpy.asarray([0,0,0,0], dtype=numpy.float64)

def fitting_function(x, a, b, c, d):
	return numpy.polyval([a, b, c, d], x)

xdata = numpy.linspace(-10,10,17)
ydata = fitting_function(xdata, *actual_params) + numpy.random.normal(size=xdata.size)

scipy_solved, _ = scipy.optimize.curve_fit(f=fitting_function, p0=initial_params, xdata=xdata, ydata=ydata)

hillclimb_solved, num_iters = hillclimb.curve_fit(f=fitting_function, p0=initial_params, xdata=xdata, ydata=ydata)

print('real parameters:', actual_params)
print('scipy-solved parameters:', scipy_solved)
print('\terror:', scipy_solved - actual_params)
print('hillclimb-solved parameters:', hillclimb_solved)
print('\terror:', hillclimb_solved - actual_params)
print('took %s iterations' % num_iters)

pyplot.plot(xdata, ydata, 'rx', label='source data')
pyplot.plot(xdata, fitting_function(xdata, *scipy_solved), 'g-', label='scipy.optimize.curve_fit')
pyplot.plot(xdata, fitting_function(xdata, *hillclimb_solved), 'b--', label='hillclimb.curve_fit')
pyplot.grid()
pyplot.legend()
pyplot.tight_layout()
pyplot.show()