# Frequently Asked Questions

## What is Maniverse intended for?

I wrote Maniverse primarily for my quantum chemistry packages
[Chinium](https://github.com/FreemanTheMaverick/Chinium)
and [Orbaplaw](https://github.com/FreemanTheMaverick/Orbaplaw),
which take care of some classic constraints in quantum chemistry via OOM.

However, Maniverse is intended for more general use than merely quantum chemistry.

## Why another library for OOM?

The two packages mentioned above are written in C++ and Python separately,
so I hoped to have a single library for both C++ and Python.

As far as I know, none of the existing libraries are 2-in-1.

## How will Maniverse be maintained?

Optimization on manifolds has two aspects:
the manifolds and the optimization algorithms.
Therefore, this question should be divided into two:
how will the two aspects be maintained separately?

For the manifolds, as a quantum chemist,
I focused more on the Stiefel manifold and
the Grassmann manifold (and their derivatives),
so major emphasis will be laid on these two.
However, users are welcomed to give advice on more manifolds to be supported.
Additionally, Maniverse provides a base class `Manifold`
from which users can derive their own manifold class.

For the optimization algorithms,
the attention is paid to the second-order methods,
because nearly all the functions to be optimized
in quantum chemistry are smooth and well-behaved.
These methods include Riemannian trust region method and Riemannian BFGS.
I would like to keep track of the popular field of OOM
and implement more efficient algorithms as they are being proposed,
as long as they enhance the performance in my projects on quantum chemistry.

### Are you an expert on OOM?
No.
It has just occurred to me that OOM can be extremely powerful
in some topics in quantum chemistry in middle 2024,
so I set out to develop Maniverse.
However, my knowledge in OOM is deficient,
and I am still learning through textbooks, papers and discussions on webs.
For helping make Maniverse real, I have a long namelist to thank.
The good thing is that the current codes do work
as they are expected, at least in my projects.