# ENE223

The objectif of this project is to develop a Fully Implicit Simulator to simulate flow in Porous Media/Reservoirs:

Before showing the equations, let’s first go over the assumptions of the developed model:
- No capillary effects.
- Oil viscosity is constant (It can be easily added to the simulator because it is treated the same way as the gas viscosity).
- Porosity is constant.
- Valid hypothesises for the well model development (Radial-flow, steady-state or pseudosteady-state, fully penetrating, No interference with boundaries or other wells).

We first formulate the equations governing flow in these media using the mass oncervation equation and Darcy's equation. Next, we use a discretization scheme IMPES to discretize these equations. We form our residual vector and from that we can compute our Jacobian and use a Newton method solving strategy.

The results from different simulations will be presented and compared to the ones from ECLIPSE software.
