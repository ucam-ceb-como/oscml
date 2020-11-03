OSCML - Organic Solar Cell Machine Learning Project
------------------------------------------------------------------------------------------
This repository contains a number of different ML models that have been developed to
predict Power Conversion Efficiency (PCE) of organic photovoltaics (OPV).


GETTING STARTED
------------------------------------------------------------------------------------------
These instructions will get you a copy of the project up and running on your local machine
for development and testing purposes.


Prerequisites
-----------------
- Anaconda installation (either Miniconda or Anaconda)
- Access to Vienna (for downloading the project data)


Installing
-----------------

Provided all above prerequisites are met the package can be installed via the following
steps:

(Windows)
1. Open Anaconda Command Prompt
2. Navigate to the project directory
3. Run:
    \> install_script.sh -a

(Linux)
1. Add Miniconda to your PATH in .bashrc by running "conda init" command.
   On HPC you may need to run "module load miniconda/3/" command first.
2. Run:
    \> conda activate
   command. This will activate your base conda environment.
3. Navigate to the project directory.
4. Run:
    \> install_script.sh -a

The steps above, regardless of the OS platform, should create a separate conda virtual
environment called "osc_env", install all required packages and download the project
data from Vienna. When downloading data, you will be prompted for your Vienna login
details. If you do not have access to Vienna, you can always obtain data from someone
who has it. Please note that data should not be version controlled.

After successful installation, please do not forget to activate the newly created
conda environment to run the code via the following command:

\> conda activate osc_env

Important note for the VS CODE users:
-----------------

It has been shown that the VS CODE and Anaconda integration is not always smooth. One
problem that users very often experience is VS CODE's failure to fully load Anaconda's
virtual environment, which in turn leads to random package import failures (e.g. numpy).
A workaround that seems to fix this issue is to always launch the VS CODE from a 
command line which has the conda environment already activated. As an example,
launching the VS CODE on Windows should be done as follows:

1. Open Anaconda Command Prompt
2. Navigate to the project directory
3. Activate the "osc_env" environment
4. Launch the VS CODE via the following command:
    \> code .


RUNNING THE TESTS
------------------------------------------------------------------------------------------
To be determined


AUTHORS
------------------------------------------------------------------------------------------
Andreas Eibeck, Daniel Nurkowski, Angiras Menon, Jiaru Bai


LICENSE
------------------------------------------------------------------------------------------
To be determined