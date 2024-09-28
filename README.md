# Differentiable Grouped Feedback Delay Networks

We proposed the Grouped Feedback Delay Network [1-3] to model multi-slope decay in late reverberation, which is commonly observed in coupled rooms and rooms with non-uniform absorption.
While the network is highly parameterised, it is still tricky to model a measured space by tuning its parameters [2]. In this work, we automatically learn the parameters of the GFDN to model
a measured coupled space. The network input and output filters are assumed to be functions of the source and listener positions, which determine the net perceived late reverberation in coupled spaces; for example: a receiver in the more reverberant space and a source in the less reverberant space will lead to two-stage decay with a fast early decay and late slow decay. If we swap the source and the listener, the net effect changes. To model this spatial distribution of the late tail, we train the input and output filters of the GFDN with deep learning. The coupled feedback matrix, on the other hand, is hypothesised to be a function of the room geometries, diffusion properties and coupling apertures, and is position-independent. This is also learnt during training with backpropagation.

The idea to use a dataset of RIRs measured in a coupled space to learn the spatial embeddings. Now, if we want to extrapolate the RIR at a new (unmeasured) position, we can do that with the
Differentiable GFDN. More powerfully, we can parameterise the late reverberation in the entire space with this very efficient network which is ideal for real-time rendering. This not only
reduces memory requirements of storing measured RIRs, but is also faster than convolution for long reverberation tails.

Clone this and modify the contents. To setup
- Create a virtual environment with `python3 -m venv .venv`. Activate it with `source .venv/bin/activate`.
- Modify `pyproject.toml` to include new libraries. Installation with pyproject.toml requires pip > 21.3
- Install the repository with `pip install -e .`
- To get setup with submodules run `git add submodule <repo-link>`, `git submodule update --init --recursive`

## Development tools

- Checks include `pylint, flake8`. 
- Unit testing with `pytest`. 
- Github action runner is `tox`. 
- To set these up, run `pip install -r dev-requirements.txt`
- To install pre-commit-hooks, run `brew install pre-commit` followed by `pre-commit install`