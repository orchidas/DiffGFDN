# Data-driven spatially-dynamic late reverberation rendering in coupled spaces for Augmented Reality

The goal of this work is to learn spatially-dynamic late reverberation properties in a complex space from a measured set of RIRs / SRIRs, and render dynamic late reverberation as the user moves around the space. Models are trained to learn from a finite set of measurements, and extrapolate late reverberation behaviour at any location in the space. To setup the repository, follow instructions in [CONTRIBUTING.md](CONTRIBUTING.md).

## Data-driven late reverberation interpolation in coupled spaces using the Common Slopes model

We investigate the modelling of position-dependent directional late reverberation in coupled spaces. We assume we have a set of <b> Spatial Room Impulse Responses (SRIRs) </b> (encoded in $N_\text{sp}$ order ambisonics) measured at several locations in the space for a fixed source-position. We want to use these to generalise the late reverb tail of the SRIRs at any point in the room. 

To do this, we leverage the <b>Common Slopes (CS)</b> model which hypothesises that the energy decay in any coupled space can be modelled as a weighted sum of decay kernels with unique reverberation times, which are position and direction-invariant. We train MLPs in octave bands to learn the weights of the decay kernel, known as the CS amplitudes in the spherical harmonics domain. Once trained, the MLPs can predict the CS amplitudes in the SH domain at any new position in space. White noise, shaped in octave bands by the predicted CS parameters, is used to synthesise the late reverberation tail. As the user navigates the space, the MLPs update the CS amplitudes, a new reverberation tail is synthesised and time-varying convolution is performed on the input signal and the synthesised late tail.

### Dataset

We have been using the dataset published [here](https://zenodo.org/records/13338346) which has three coupled rooms simulated with Treble's hybrid solver and has 2nd order ambisonic SRIRs at 838 receiver locations for a single source location. This has been saved in the path `resources/Georg_3Room_FDTD/`. To parse the dataset and save the SRIRs and CS parameters in octave bands, run `python3 src/convert_mat_to_pkl_ambi.py`

### Training

The scripts for training this model are in the [src/spatial_sampling](src/spatial_sampling/) folder. 
- To run training on the three coupled room dataset, you can run the script [src/run_spatial_sampling_test.py](src/run_spatial_sampling_test.py). The MLPs are trained on a particular frequency band, an example of a config file is available at [here](data/config/spatial_sampling/treble_data_grid_training_500Hz_directional_spatial_sampling_test.yml). TLDR; to run training, run `python3 src/run_spatial_sampling_test -c <config_path>`
- Once trained, you can run inferencing with `python3 src/run_spatial_sampling_test -c <config_path> --infer` which will plot the results.
- To generate synthetic SRIR tails once all octave bands have been trained, you can use functions in the script [src/spatial_sampling/inference.py](src/spatial_sampling/inference.py). An example of how to use this script to generate binaural sound examples for moving listeners has been provided in [this notebook](notebooks/create_binaural_sound_examples.ipynb).


## Differentiable Grouped Feedback Delay Networks for data-driven late reverberation rendering in coupled spaces

We proposed the [Grouped Feedback Delay Network](https://github.com/orchidas/GFDN) to render multi-slope late reverberation. In this work, we automatically learn the parameters of the GFDN to model multi-slope reverberation in a complex space from a set of measured <b>Room Impulse Responses</b>. 


A dataset of RIRs measured in a coupled space, along with the corresponding source and receiver positions, can be used to train the DiffGFDN. Now, the RIR at a new (unmeasured) position can be extrapolated with the DiffGFDN network. More powerfully, we can parameterise the late reverberation in the entire space with a very efficient network which is ideal for real-time rendering as the source-listener moves. This not only
reduces memory requirements of storing measured RIRs, but is also faster than convolution with long reverberation tails.

### Dataset

We use omni-directional and spatial RIRs from the same three-coupled room dataset. The mat files are converted to pickle files (for faster loading) and filtered in octave bands using the scripts `python3 src/convert_mat_to_pkl.py` and `python3 src/convert_mat_to_pkl_ambi.py` respectively.

<!--Additionally, we have tools for generating a synthetic dataset of coupled rooms by shaping noise (see [gdalsanto/slope2noise](https://github.com/gdalsanto/slope2noise/blob/main/config/rir_synthesis_coupled_room.yml)). 

To generate your own synthetic datasets,
- Set up the submodules (see [CONTRIBUTING.md](CONTRIBUTING.md)). 
- Navigate to `submodules/slope2rir` and run `python3 main.py -c <config_path>`. An example of a config file to generate a coupled room dataset is available [here]( submodules/slope2rir/config/rir_synthesis_coupled_room_single_batch.yml).

To work with the files that we have tested on, use `git lfs`.
- Install with `brew install git lfs`
- Go to repo, and run `git lfs install`
- Add the appropriate file with `git lfs track <filepath>`
- Add and commit the file. Push it to origin.
- To download files tracked with LFS, run `git lfs pull origin <branch_name>` -->

To use an open-source dataset:
- Upcoming, not implemented yet!

### Training

- Omnidirectional DiffGFDN
	- To run training of a single full-band GFDN on a grid of receiver positions, create a different config file (example [here](./data/config/treble_data_grid_training_full_band_colorless_loss.yml)). Then run `python3 src/run_model.py -c <config_file_path>`. 
	- To run training with one frequency-independent omni-directional DiffGFDN for each octave band, create config files for each band, and run the training for each config file. Alternately, run `python3 src/run_subband_training_treble.py --freqs <list_of_octave_frequencies>`
	- To only run inference on the trained parallel octave-band GFDNs, run `python3 src/run_subband_training_treble.py`. This will save the synthesised RIRs as a pkl file.
- Directional DiffGFDN
	- To run training on a single frequency band, create a config file (example [here](./data/config/directional_fdn/treble_data_grid_training_1000Hz_directional_fdn_grid_res=0.6m.yml))
	- After traning all frequency bands, inference can be run using `infer_all_octave_bands_directional_fdn` in `diff_gfdn.inference.py`.


## Publications

- <i>Neural-network based interpolation of late reverberation in coupled spaces using the common slopes model</i> - Das, Dal Santo, Schlecht and Zvetkovic, submitted to IEEE Work. Appl. of Sig. Process. Aud. Acous., IEEE WASPAA 2025.
- <i>Differentiable Grouped Feedback Delay Networks: Learning from measured Room Impulse Responses for spatially dynamic late reverberation rendering</i> - Das, Dal Santo, Schlecht and Zvetkovic, submitted to IEEE Trans. Aud. Speech Lang. Process., IEEE TASLP, 2025.
- <i> Differentiable Grouped Feedback Delay Networks for Learning Position and Direction-Dependent Late Reverberation</i> - Das, Schlecht, Dal Santo, Cvetkovic, submitted to IEEE Int. Conf. Aud., Speech Sig. Process., ICASSP 2026.



## Sound examples

Mono sound examples of the DiffGFDN are available [here](https://ccrma.stanford.edu/~orchi/FDN/GFDN/DiffGFDN/). Binaural sound examples of the DiffGFDN are available [here](https://ccrma.stanford.edu/∼orchi/FDN/GFDN/DiffGFDN/ICASSP26/). Binaural sound examples of convolution-based directional rendering are available [here](https://ccrma.stanford.edu/∼orchi/FDN/GFDN/DiffGFDN/WASPAA25/). 
