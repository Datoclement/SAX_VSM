Guide of Code:

    0. Pre-compilation: go into _python folder and run "python setup.py build_ext --inplace" to compile Cython file

    1. How to Make a Quick Run: "python test.py"

    2. Structure of the code:

        0. "data" contains all the data used, and "output" is an empty folder to hold potential output

        1. There are 13 files from the _python:

            #DOCUMENTATION
            0. README.md

            #CORE
            1. saxvsm.py
                Implementation of SAXVSM classifier as a sklearn estimator
                Performance-deciding processes are moved to performance_critical_functions.pyx
            2. optimizer.py
                Opimisation of hyper-parameters of SAXVSM

            #CYTHON
            3. performance_critical_functions.pyx
                Heavily used processes are implemented in Cython to take the advantage of C efficiency
            4. setup.py
                Basic file to allow Cython to be compiled

            #HANDY FUNCTIONS
            5. parser.py
                Functions to read data from uts_dataset and mts_dataset
            6. formats.py
                Functions to provide pretty printing
            7. helper.py
                Frequently used functions for vectorized operations

            #MAIN
            8. test.py
                Most of the usage is demonstrated.

            #MODULE
            9. __init__.py

            #DEPRECIATED
            (These include an attempt of bagging ensemble in varying hyper-parameters of SAXVSM and an attempt of improving weights of words in SAXVSM by Stochastic Gradient Descent, which appear to be not especially interesting after some initial experiments)
            10. ensemble.py
            11. gradient_descent.py
            12. sparse_matrix.py (used for SGD)
