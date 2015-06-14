Introduction
############

The Material Model Laboratory (Matmodlab) is a suite of tools whose
purpose is to aid in the rapid development and testing of material models.
Matmodlab is made up of several components, the most notable being the
Material Model Driver. The material model driver can be thought to drive a
single material point of a finite element simulation through very specific
user designed paths. This permits exercising material models in ways not
possible in finite element calculations, desgining verification and validation
tests of the material response, among others. Matmodlab is a small suite
of tools at the developers disposal to aid in the design and implementation of
material models in larger finite element host codes. It is also a useful tool
to analysists for understanding and parameterizing a material's response to
deformation.

The core of Matmodlab code base is written in Python and leverages
Python's object oriented programming (OOP) design. OOP techniques are used
throughout Matmodlab to setup and manage simulation data. Computationally
heavy portions of the code, and many material models themselves are written in
Fortran for its speed and ubiquity in scientific computing. Calling Fortran
procedures from Python is made possible by the ``f2py`` module, standard in
Numpy, that compiles and creates Python shared object libraries from Fortran
sources.

Output files from Matmodlab simulations are in the *dbx* or database
ExodusII format. Since Matmodlab is designed to be used by material model
developers, it is expected that the typical user will want access to *all*
available output from a material model, thus all simulation data is written to
the output database. Output files can be visualized with the Matmodlab
visualization utility.

Matmodlab is free software released under the MIT License.

.. toctree::
   :hidden:

   bg
   quickstart
   solmeth
   cli
