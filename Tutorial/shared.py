import theano.tensor as T
from theano import function, shared

count = shared(0)
inc = T.iscalar('inc')
accum = function([inc], count, updates=[(count, count + inc)])

#initial value is 0
print count.get_value()

# increment it by 1. current value 0 is printed.
print accum(1)

# check the new value which is 1.
print count.get_value()

# increment by 300. current value 1 is printed.
print accum(300)

# print the new value 301
print count.get_value()

#set counter to -1
count.set_value(-1)

# increment by 3, current value -1 is printed
print accum(3)

# counter's new value is -1 + 3 = 2
print count.get_value()


