# Maniverse
> Numerical optimization on manifolds

## Overview
Maniverse is a library for optimization on manifolds (OOM).

### What is Maniverse intended for?
I wrote Maniverse primarily for my quantum chemistry packages [Chinium](https://github.com/FreemanTheMaverick/Chinium) and [Orbaplaw](https://github.com/FreemanTheMaverick/Orbaplaw), which take care of some classic constraints in quantum chemistry via OOM.
However, Maniverse is intended for more general use than merely quantum chemistry.

### Why another library for OOM?
The two packages mentioned above are written in C++ and Python separately, so I hoped to have a single library for both C++ and Python.
As far as I know, none of the existing libraries are 2-in-1.

### How will Maniverse be maintained?
Optimization on manifolds has two aspects: the manifolds and the optimization algorithms.
Therefore, this question should be divided into two: how will the two aspects be maintained separately?

For the manifolds, as a quantum chemist, I focused more on the Stiefel manifold and the Grassmann manifold (and their derivatives), so major emphasis will be laid on these two.
However, users are welcomed to give advice on more manifolds to be supported.
Additionally, Maniverse provides a base class `Manifold` from which users can derive their own manifold class.

For the optimization algorithms, the attention is paid to the second-order methods, because nearly all the functions to be optimized in quantum chemistry are smooth and well-behaved.
These methods include Riemannian trust region method and Riemannian BFGS.
I would like to keep track of the popular field of OOM and implement more efficient algorithms as they are being proposed, as long as they enhance the performance in my projects on quantum chemistry.

### Are you an expert on OOM?
No.
It has just occurred to me that OOM can be extremely powerful in some topics in quantum chemistry in middle 2024, so I set out to develop Maniverse.
However, my knowledge in OOM is deficient, and I am still learning through textbooks, papers and discussions on webs.
For helping make Maniverse real, I have a long namelist to thank.
The good thing is that the current codes do work as they are expected, at least in my projects.

## Prerequisites
* A C++ compiler that supports C++17 standard
* GNU make
* [Eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page) >= 3.4.90
* [PyBind11](https://pybind11.readthedocs.io/en/stable/index.html#) >= 2.13.6 (For interface to python)
* Python3 with numpy (For interface to python)

## Installation
### Manual build
* Cloning the repository
```
$ git clone https://github.com/FreemanTheMaverick/Maniverse.git
```
* Edit the first few lines of `/Maniverse/makefile` for your own computer configuration, including
  * the commands that call the C++ compiler, the GNU make and `ar`
  * the option that indicates whether to build for C++ use or python use
  * the directories that contain the necessary libraries
* ```make -j[N]``` and you will find the newly created directories `/Maniverse/include` and `/Maniverse/lib`.
* Utilize Maniverse in your project
  * For C++,
    ```
    $ g++ test.cpp -isystem $(MANIVERSE)/include/ -L$(MANIVERSE)/lib/ -l:libmaniverse.a # Static linking
    $ g++ test.cpp -isystem $(MANIVERSE)/include/ -L$(MANIVERSE)/lib/ -lmaniverse # Shared linking
    ```
  * For Python,
    ```
    $ export PYTHONPATH=$PYTHONPATH:$(MANIVERSE)/lib/
    $ python
    >>> import Maniverse as mv
    ```
### Pip (for Python use only)
* Setting environment variables
```
$ export PYTHON3=[The path where you can find "Python.h".] # You may check this by the command "locate Python.h".
$ export EIGEN3=[The path where you can find "Eigen/", "signature_of_eigen3_matrix_library" and "unsupported/".] # This is optional. If this is not set, the Eigen3 library will be downloaded automatically.
```
* Installation with `pip`
```
pip install Maniverse
```
Usually `pip` installs packages to a `lib/` directory that is already in `$PYTHONPATH`, so you do not need to set the environment variable for Maniverse.
* Utilize Maniverse in your project
```
$ python
>>> import Maniverse as mv
```
