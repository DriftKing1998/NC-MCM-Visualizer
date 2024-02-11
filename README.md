# NC-MCM-Visualizer 

## A toolbox to visualize neuronal imaging data and apply the NC-MCM framework to it

This is a toolbox uses neuronal & behavioral data and visualizes it. The main functionalities include: 
- creating different diagnostic plots and fitting a models on the data
- clustering datapoints into cognitive clusters using behavioral probability trajectories 
- testing and plotting probability of the cognitive sequence being a markov process of 1st order
- creating 3D visualizations using different sklearn dimensionality reduction algorithms as mappings
- the possibility to create a neural manifold using custom BunDLeNet's or any other mapping added
- creating movies and plots of behavioral/neuronal trajectories using the 3D mapping

### These are some of the plots created from calcium imaging data of C. elegans
#### Mean probability of being a 1st order markov process for all 5 worms at different amounts of cognitive states (30 reps)
<img src="data/plots/Demonstration/AverageMarkov.png" width="700" alt="Mean probability to be a 1st order markov process for all worms">

#### Behavioral state diagram for worm 3 and 3 cognitive states
<img src="data/plots/Demonstration/NormalPlot.png" width="700" alt="Behavioral State Diagram for Worm 3 and 3 cognitive states">

#### Interactive behavioral state diagram for worm 3 and 3 cognitive states (saved as a .html file)
<img src="data/plots/Demonstration/InteractivePlot.png" width="700" alt="Behavioral State Diagram for Worm 3 and 3 cognitive states - interactive">

#### Behavioral state adjacancy matrix for worm 3 and 3 cognitive states
<img src="data/plots/Demonstration/AdjacancyMatrix.png" width="600" alt="Behavioral State Diagram for Worm 3 and 3 cognitive states - adjancency matrix">

#### Comparison of predicted and true label using BunDLeNet's tau model as mapping and its predictor on worm 3
<img src="data/plots/Demonstration/ComaprisonBunDLeNet.png" width="700" alt="Comparison between true and predicted label using BunDLeNet as mapping and predictor">

#### Movie using BunDLeNet's tau model as mapping on worm 1
<img src="data/plots/Demonstration/Worm_1_Interval_100.gif" width="700" alt="Movie using BunDLeNet's tau model as mapping and the true labels">

## Installation and usage information (for end-users)
To get the toolbox running on your own PC I still need to do this:

Create a Python Package: First, create your Python package with the necessary code and structure. This typically involves organizing your code into a package structure (with __init__.py files) and possibly writing a setup.py file for distribution purposes.

Publish on GitHub: Host your package's code on GitHub. Make sure your repository is public so that anyone can access it.

PyPI Setup: Create an account on PyPI if you don't have one already. You'll need to create a setup.py file for your project. This file contains metadata about your package, including its name, version, and dependencies. You can find examples and detailed documentation on the Python Packaging Authority's website.

Create a Release: Tag a release on GitHub. This typically involves creating a release in your GitHub repository and attaching the source distribution (a .tar.gz file) generated by setup.py sdist.

Upload to PyPI: Use a tool like twine to upload your package to PyPI. Install twine if you haven't already (pip install twine), then run twine upload dist/* from the directory containing your source distribution files.

Installation: After you've uploaded your package to PyPI, users can install it with pip install your-package-name.

Importing: Once installed, users can import your package in their Python code using import your_package_name.

## Installation and usage information (for contributors)