# Installation

## Install conda (miniconda)
The first thing we have to do is install conda so that we can later install mamba (basically a faster version of conda). We have limited space, so we are going to download miniconda which gives you access to the most basic conda functionalities, without taking up loads of space like conda typically does.

Access the miniconda installer [here](https://docs.conda.io/en/latest/miniconda.html#linux-installers). The installer you download will be specific to the operating system you're running, so in my case, I went to the Linux Installers subsection and downloaded the first installer on the list (Miniconda3 Linux 64-bit). You can download the installer by running the following code:

`wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh`


Great, you've downloaded the conda installer. Now all you have to do is run the installer by entering the following into the command line:

`bash Miniconda3-py310_23.1.0-1-Linux-x86_64.sh`

You'll have to press enter a bunch of time, then it'll ask you to accept some kind of agreement. Agree to all of this, then, when it asks for a path, put whichever directory you want all of your conda related files to go into. In my case, my home directory is packed, so I've just placed this in my directory of the octdata2 data folder.

The path I entered was '/autofs/cluster/octdata3/users/epc28/miniconda'. This installer will create this directory for you, so do not make it via the mkdir command, or else it will be very angry at you.

Wait for miniconda to install. When it asks you if it should run the init, just agree and your life will be 100x easier.

## Install mamba (micromamba)
I've installed mamba to my directory in octdata2 by then running: 

`conda install mamba -c conda-forge -n base`

This is a good time for a coffee break.

## Create vesselsynth environment

To create the environment for vesselsynth, run this:

`mamba create -n vesselsynth -c conda-forge -c pytorch python=3.10.9 ipython pytorch cudatoolkit=11.7 cupy cppyy -r micromamba`


## Activate vesselsynth
Sweet, now activate that bad larry!

`mamba activate vesselsynth`

## Installing stuff with pip

We need to install some more of Yael's code, and we're going to do that via pip so we can just import the models from our virtual environment. Paste this whole thing into your terminal.

`pip install git+https://github.com/balbasty/cornucopia git+https://github.com/balbasty/jitfields.git@6076f915f6556c9733958a0aab28ed7ee93301e8 git+https://github.com/balbasty/torch-interpol git+https://github.com/balbasty/torch-distmap`

# Vesselsynth

## Project organization

I've included the project tree below just so you know what's going on. This tree is not exhaustive, but it is representitive. I've really only included the things you need to worry about. Don't worry about the data folder for now, I'll explain that later.

```
.
├── data
│   ├── exp0001
│   └── exp*
├── README.md
├── scripts
│   └── vessels_oct.py
├── setup.cfg
├── setup.py
├── torch.yaml
└── vesselsynth
```

## Generating data
Great. Now you're ready to start making synthetic data! To do this, simply change your working directory to the base of vesselsynth. For me, this means running the following command in the terminal:

`cd /autofs/cluster/octdata2/users/epc28/vesselsynth`

Then, run the actual synthesis code:

`python3 scripts/vessels_oct.py`

Now's a good time for another coffee break. When you get back, make sure the script is running by opening a new window in your terminal and entering `ps -ef | grep python3` or just looking at the output of the `top` command. The lazy man's way of doing this (the way i do this) is by putting my hand on my computer... Hmmmm - toasty!

## Viewing data

That script you just ran should have created a new folder in your vesselsynth base directory called "data". This directory is home to all of your experimental runs. The data produced from each run is contained in its own subdirectory within this data folder - in the form of "exp*". Your most recent run should be the largest number on the screen. If this was your first time running the script, you should only see a folder named "exp0001".


# Vesselseg

## Project organization

The project tree is shown below. Again, I've really only included the most pertinent things

```
.
├── epctrain.py
├── output
│   ├── output
│   ├── tb_logs
│   └── test
├── README.md
├── scripts
│   ├── epc-retrain.py
│   ├── test_synth.py
│   ├── train_synth.py
│   └── vessel_synth.py
└── vesselseg
```

## 