Notes
=====

Notes on the manuscript.

Introduction
------------

Ideas:

- [ ] Dust is important for ISM, star formation, protoplanetary discs, debris discs.
- [ ] Dust makes planets.
- [ ] Dust regulates temperature of disc.
- [ ] Dust is observable.
- [x] Multi-wavelength observations show that the radial extent of emission scales inversely with wavelength. I.e. larger grains drift inwards more \citet{Andrews2012ApJ...744..162A}. Cite \citet{Testi2014prpl.conf..339T}?
- [x] Dust-gas modeling in astrophysical fluids.
- [x] \citet{Haworth2016PASA...33...53H}: grand challenges in protoplanetary disc modeling.
- [x] Applications: protoplanetary discs, debris discs.

Motivations for multigrain:

- [ ] coupling with MCFOST
- [ ] self-gravitating discs
- [ ] time scale for gap opening

Methods
-------

### Drag timescale

- [ ] Differentiate between Epstein and Stokes regimes?
- [ ] Mention terminal velocity approximation?

### SPH with multiple dust species

- [ ] Mention use of quintic kernel?
- [ ] Show details of drag kernel?
- [ ] Mention use of gas smoothing length in drag kernel.

Numerical tests
---------------

- [ ] Add particular version of Phantom specified by git commit hash.

### Dusty box

- [x] alpha = 0, beta = 2

### Dusty wave

- [x] alpha = 0, beta = 2

### Dusty shock

- [ ] Add number of particles.
- [x] alpha = 1, beta = 2
- [ ] Add parameter table?

Discussion
----------

- [x] Time step constraint via Stokes number. I.e. only appropriate for large grains.
- [x] Memory constraint: extra set of particles per dust species requires position, velocity, etc. unlike the mixture method.
- [x] Dustyshock hfact comparison; see Figure~\ref{fig:dustyshock_hfact}.
- [x] Dust particles don't rearrange like gas particles (due to the lack of pressure gradient force) and that requires summing over more neighbours (i.e. larger `hfact`).
- [x] Using a larger `hfact` for the dust compared with the gas. This is a novel idea.
- [ ] The pressureless fluid approximation - crossing trajectories.
- [ ] Ignore: grain growth; radiation effects.

Conclusions
-----------

- [x] Implemented SPH numerical scheme for multiple dust species represented by separate sets of SPH particles, appropriate for large Stokes number regime.
- [x] Demonstrated that the method is accurate by testing on known problems.
- [x] Suggest that `hfact` should be larger for dust compared to gas.

Acknowledgements
----------------

- [x] Add IPython and Jupyter.

Appendix
--------

- [ ] Reproducibility: github/zenodo for Phantom config files.
