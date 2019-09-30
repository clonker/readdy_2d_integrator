import numpy as np
print('1')
import readdy
print('2')
import readdyextension as ext
print('3')

system = readdy.ReactionDiffusionSystem([10, 10, 10])
system.add_species("A", 0.001)
custom_integrator = ext.CustomIntegrator(.1)
sim = system.simulation(kernel="CPU")
sim.integrator = custom_integrator

sim.add_particles("A", np.random.random((3000, 3)))
sim.run(10000, .1)
