import sys
sys.path.append('/home/nm13850/Documents/PhD/python_v2/Nick_functions')

# import hello_world              #works
# hello_world.hello_world()

# import hello_world as hw              #works
# hw.hello_world()

from hello_world import hello_world  # best
hello_world()

from test_func import test_func1, test_func2
test_func1() #Calling function defined in add module.
test_func2()