## To setup

- Create a virtual environment with `python3 -m venv .venv`. Activate it with `source .venv/bin/activate`.
- Modify `pyproject.toml` to include new libraries. Installation with pyproject.toml requires pip > 21.3
- Install the repository with `pip install -e .`
- To get setup with submodules run `git submodule add <repo-link>`, `git submodule update --init --recursive`
- Manually install the submodule with `pip install -e submodules/slope2noise`
- To run accelerated training on Apple silicon (requires conda), install pytorch nightly build with `pip3 install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu`

## Development tools

- Checks include `pylint, flake8`. 
- Unit testing with `pytest`. 
- Github action runner is `tox`. 
- To set these up, run `pip install -r dev-requirements.txt`
- To install pre-commit-hooks, run `brew install pre-commit` followed by `pre-commit install`