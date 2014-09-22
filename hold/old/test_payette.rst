
#############
Running Tests
#############

*Payette* provides a mechanism for running material model regression tests with
the :file:`testPayette` script. :file:`testPayette` is what makes *Payette* an
invaluable tool to material model development. Simplistically,
:file:`testPayette` is a tool that exercises :file:`runPayette` on previously
setup simulations and compares the output against a "gold" result, allowing model
developers to instantly see if and how changes to a material model effect its
output.


Basic Usage of :file:`testPayette`
==================================

From a command prompt execute::

  % testPayette

:file:`testPayette` will scan :file:`PAYETTE_ROOT/Tests` for tests. After
creating a lists of tests to run, :file:`testPayette` will create a
:file:`PWD/TestResults.OSTYPE` directory and subdirectories in which the tests
will be run. The subdirectories created in :file:`PWD/TestResults.OSTYPE/` mirror
the subdirectories of :file:`PAYETTE_ROOT/Tests`. After creating the
:file:`PWD/TestResults.OSTYPE/` directory tree, it will then move in to the
appropriate subdirectory to run and analyze each individual test. For example, if
the :file:`Tests/Materials/Elastic/elastic-unistrain.py` test is run,
:file:`testPayette` will create a
:file:`PWD/TestResults.OSTYPE/Materials/Elastic/elastic-unistrain/` directory,
move to it, and run the :file:`elastic-unistrain.py` test in it.


Filtering Tests to Run
----------------------

The tests run can be filtered by keyword with the ``-k`` option. For example, to
run only the ``fast`` ``kayenta`` material tests, execute::

  % testPayette -k kayenta -k fast

Or, you can run specific tests with the ``-t`` option. For example, to run only
the :file:`elastic-unistrain.py` test, execute::

  % testPayette -t elastic-unistrain


Multiprocessor Support
======================

By default, :file:`testPayette` will run all collected tests serially on a single
processor, but supports running tests on multiple processors with the ``-j
nproc`` option, where ``nproc`` is the number of processors. For example, to run
all of the kayenta tests on 8 processors, execute::

  % testPayette -k kayenta -j8

.. warning::

   When run on multiple processors, you cannot kill :file:`testPayette` by
   ``ctrl-c``, but must put the job in the background and then kill it. Any help
   from a python multiprocessor expert on getting around this limitation would be
   greatly appreciated!


Other :file:`testPayette` Options
=================================

Execute::

  % testPayette -h

for a complete list of options.


html Test Summary
=================

:file:`testPayette` gives a complete summary of test results in
:file:`TestResults.OSTYPE/summary.html` viewable in any web browser.


Mathematca Post Processing
==========================

For the Kayenta material model, :file:`testPayette` creates Rebecca Brannon's
Mathematica post processing file :file:`kayenta.nb` (formerly :file:`030626.nb`)
in :file:`PWD/TestResults.OSTYPE/Materials/Kayenta/`, that aids greatly in
analyzing test results for the Kayenta material model.


Creating New Tests
==================

.. todo::

   need to add this section

