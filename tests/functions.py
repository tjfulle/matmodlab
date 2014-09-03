#!/usr/bin/env mmd

K = 10.e9
G = 3.75e9

a = np.array([[0., 2.], [1., 3.], [2., 4.]])
f2 = Function(2, "analytic_expression", lambda t: np.sin(t))
f3 = Function(3, "piecewise_linear", a)
functions = [f2, f3]

path = """
{0} 2:1.e-1 0 0
""".format(2*pi)

# set up the driver
driver = Driver("Continuum", path=path, path_input="function",
                num_steps=200, functions=functions, cfmt="222")

# set up the material
parameters = {"K":K, "G":G}
material = Material("elastic", parameters=parameters)

# set up and run the model
runid = "functions"
mps = MaterialPointSimulator(runid, driver, material)
mps.run()

mps.extract_from_db(variables=["STRESS", "STRAIN"], format="ascii",
                    step=1, ffmt="12.6E")
mps.visualize_results()
