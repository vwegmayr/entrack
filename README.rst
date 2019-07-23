=====================================================================
Entrack: A Data-Driven Maximum-Entropy Approach to Fiber Tractography
=====================================================================

.. _sumatra: https://pythonhosted.org/Sumatra/
.. _miniconda: https://conda.io/docs/install/quick.html
.. _`example config`: .example_config.yaml
.. _VirtualBox: https://www.virtualbox.org/
.. _Ubuntu: https://www.ubuntu.com/download/desktop
.. _runner: run.py
.. _models: ml_project/modules/models
.. _`.environment`: .environment
.. _Tractometer: http://www.tractometer.org/
.. _Trackvis: http://www.trackvis.org/
.. _`Human Connectome Project`: http://www.humanconnectomeproject.org
.. _`MRTrix`: http://www.mrtrix.org/

This repository contains the code for the *Entrack* model and framework, as described in the paper
"Entrack: A Data-Driven Maximum-Entropy Approach to Fiber Tractography".

.. contents::


Purpose
=======

Data-driven tractography poses several different challenges to the researchers
approaching the topic for the first time. To begin with, although tractography
in general could be now considered a well-investigated field and has been
actively  studied for decades, its declination as a learning problem is very
recent and there is little work readily available. Moreover, the impossibility
to produce gold-standard tractograms renders the idea of "learning from
examples" very complex and prone to epistemological errors. On top of this
fundamental obstacles, the practical difficulties that arise when dealing with
optimization problms involving "images" that are effectively multi-dimensional
vectors in a 3D matrix can seem overwhelming.

In the paper accompanying this repository we addressed all these problems, while
also trying to build a framework in which supervised fiber tractography with
neural networks can be carried out with ease, focusing on the machine-learning
aspects rather than on the complex support machinery.


Getting Started
===============

Get Started (Non-Linux)
-----------------------

The project framework has been tested mainly in Linux (Ubuntu) environments. If
you are using Linux already, you can skip forward to Get Started for Linux.
The framework should also work on OS X, but it has not been tested extensively.
OS X users may choose to skip forward to Get started for Linux and OS X.
If you are using Windows, you need to install VirtualBox_ and create an 64-bit
Ubuntu_ virtual machine (VM).

Make sure you allocate sufficient RAM (>= 8GB) and disk space (>= 64GB) for the
VM. If you can not choose 64-bit Ubuntu in VirtualBox, you might have to enable
virtualization in your BIOS.
Once your VM is running, open a terminal and install git:

.. code-block:: shell

    sudo apt-get install git

After that, please continue with Getting Started for Linux.

Get Started (Linux and OS X)
----------------------------

First you need to install miniconda_ on your system. If you already have Anaconda
installed you can skip this step.

Having installed miniconda, clone the repository and run the setup script:

.. code-block:: shell

    git clone https://gitlab.vis.ethz.ch/vwegmayr/ml-project.git
    cd ml-project
    python setup.py --all

This will setup the :code:`entrack` environment with all the dependencies and
the sumatra_ database for experiment management.

To use some of the more advanced functionalities of the framework, you'll also
need to setup the Tractometer_ tool. This will enable the scoring of produced
fibers against this anathomically-precise ground-truth. To do this, run
additionally:

.. code-block:: shell

    python setup.py --tractometer --tm_data_dir DATA_DIR

specifying in :code:`DATA_DIR` the directory in which you would like the data
for the Tractometer to be installed in. Be aware that the size of the data is
above 700 MB. This setup script will install an additional python 2 conda
environment called :code:`entrack_tm` for the dependencies of the Tractometer.

Get the data
------------

Many different data types can be used and experimented with using this
framework. For the representation of fibers, we used the Trackvis_ format.
Therefore, the fibers that have to be loaded in the training process and the
fibers produced at prediction step are stored in :code:`.trk` files.

For a detailed explaination on the use of such data and on the assumptions made,
please refer to the paper.

Redarding the actual diffusion data, any representation can be used. This
usually comes in :code:`nii` format. In our exeperiments, we used  *Spherical
Harmonics Coefficients* as the main representaiton for the diffusion data.

The main source for diffusion datasets was the `Human Connectome Project`_. The
*teacher fibers* (i.e. the labels for the model) were produced mainly using the
iFOD2 algorithm implementation in MRTrix_.

Experiments
===========

Implemented models
------------------

In the modules_ folder you will find all the basis for the framework. The base
classess for tracking, :code:`ProbabilisticTracker` and
:code:`DeterministicTracker`, are already implemented. These classes capture two
different ways of building stramlines: either by deterministic extension or by
sampling a probability distribution. Two fully implemented models are moreover
present. The first is the :code:`SimpleTracker`, a deterministic tractography
algorithm. The second is the :code:`MaximumEntropyTracker`, which is the
probabilistic, maximum-entropy regularized model described in the paper. Refer
to it for more infomation about this model.

One last model :code:`BayesianTracker` is implemented, but it is not complete
and will be the object of future work.

Writing your own models
-----------------------

Derive your models from the :code:`ProbabilisticTracker` or
:code:`DeterministicTracker`, or, if more freedom is needed, from
:code:`BaseTracker`.

All the models need to implement, as a minimal requirement, the :code:`model_fn`
function that describes the inputs, outputs and architecture of the network.
