from theano import *
import theano.tensor as T

# a and b are double precision matrices (instantiated at once).
a, b = T.dmatrices('a', 'b')

# define three functions based on a and b
diff = a - b
abs_diff = abs(diff)
diff_squared = diff**2

# define compiled function that computs all three in one shot.
f = function([a, b], [diff, abs_diff, diff_squared])

# excecution
y = f([[1, 1], [1, 1]], [[0, 1], [2, 3]])

for a in y:
    print a

