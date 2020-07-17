# Introduction

Ideas:

- Dust is important for ISM, star formation, protoplanetary discs, debris discs.
- Dust makes planets.
- Dust regulates temperature of disc.
- Dust is observable.
- Multi-wavelength observations show that the radial extent of emission scales inversely with wavelength. I.e. larger grains drift inwards more \citet{Andrews2012ApJ...744..162A}. Cite \citet{Testi2014prpl.conf..339T}?
- Dust-gas modeling in astrophysical fluids.
- \citet{Haworth2016PASA...33...53H}: grand challenges in protoplanetary disc modeling.
- Single grain size vs multiple dust species.
- Phantom dust methods: \citet{Price2018PASA...35...31P}.
- Phantom 1-fluid multigrain: \citet{Hutchison2018MNRAS.476.2186H}.
- Applications: protoplanetary discs, debris discs.

Motivations for multigrain:

- circumbinary discs
- outward migration of dust
- backreaction
- spectral index maps
- coupling with MCFOST
- self-gravitating discs
- time scale for gap opening

# Methods

## Drag timescale

- Differentiate between Epstein and Stokes regimes?
- Mention terminal velocity approximation?

## SPH with multiple dust species

- Mention use of quintic kernel?
- Show details of drag kernel?
- Mention use of gas smoothing length in drag kernel.

# Numerical tests

- We implemented the above scheme in Phantom...
- Each test is performed in 3d.
- Quintic kernel.
- Numerical parameters including: hfact, Courant, force dt.
- Add particular version of Phantom specified by git commit hash.

## Dusty shock

Equation of state -- adiabatic with gamma = 1.

# Discussion

- Time step constraint via Stokes number. I.e. only appropriate for large grains.
- Memory constraint: extra set of particles per dust species requires position, velocity, etc. unlike the mixture method.
- Dustyshock hfact comparison; see Figure~\ref{fig:dustyshock_hfact}.

# Conclusions

- Implemented SPH numerical scheme for multiple dust species represented by separate sets of SPH particles, appropriate for large Stokes number regime.
- Demonstrated that the method is accurate by testing on known problems.
- Applied the method to simulate physical processes in proto-planetary discs.
