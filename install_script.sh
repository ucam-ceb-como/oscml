#!/bin/bash
# D. Nurkowski (danieln@cmclinnovations.com)


AUTHOR="Daniel Nurkowski <danieln@cmclinnovations.com>"
SPATH="$( cd  "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/"

CONDA_ENV_NAME="osc_env"
DATA_LOCAL="./data/raw"
DATA_REMOTE="vienna.cheng.cam.ac.uk:/home/userspace/CoMoCommon/Ongoing/Projects/c4e-jps-OSC/Data/Raw/*.*"

function check_conda {
    echo "1. Verifying conda installation."
    conda_version=$(conda --version)
    if [ $? -eq 0 ]; then
        echo "Found "$conda_version
    else
        echo "Couldn't find conda installation. On Windows, you must run this script from Anaconda Prompt for conda to be correctly located. Aborting installation."
        exit -1
    fi
}

function recreate_conda_env {
    echo "2. Recreating conda project" $CONDA_ENV_NAME "environment (in the default conda environment directory) and installing all necessary dependencies."
    # firstly remove the environment
    conda remove -y --name $CONDA_ENV_NAME --all
    # update the environment name in the yaml file
    sed -i '1s/.*/name: '$CONDA_ENV_NAME'/' environment.yml
    # create the environment
    conda env create -f environment.yml
}

function get_data_from_server {
    echo "3. Downloading required project data from the server..."
    if [ -d $DATA_LOCAL ] 
    then
        echo "Directory" $DATA_LOCAL "already exists. Remove it if you wish to re-download data from the server."
    else
        mkdir -p $DATA_LOCAL
        echo -n "Please provide your Vienna username: "
        read USERNM
        scp $USERNM"@"$DATA_REMOTE $DATA_LOCAL"/."
    fi
}

function install_ZhouLiML_package {
    echo "4. Installing the project as a python package..."
    source activate $CONDA_ENV_NAME

    if [ $? -eq 0 ]; then
        pip install -e .
    else
        echo "Couldn't activate conda" $CONDA_ENV_NAME "environment. Aborting." 
        exit-1
    fi
}

if [ "$1" == "-e" ]
then
    check_conda
    recreate_conda_env
elif [ "$1" == "-i" ]
then
    check_conda
    install_ZhouLiML_package
elif [ "$1" == "-d" ]
then
    get_data_from_server
elif [ "$1" == "-a" ]
then
    check_conda
    recreate_conda_env
    install_ZhouLiML_package
    get_data_from_server
else
    echo "==============================================================================================================="
    echo "OSCML project installation script."
    echo
    echo "Please run the script with one of the following flags set:"
    echo "--------------------------------------------------------------------------------------------------------"
    echo "  -a  : performs all the steps below"
    echo "  -e  : re-creates conda environment for this project; it will remove the current '"$CONDA_ENV_NAME"'"
    echo "        environment, if exists, and create it again installing all necessary packages"
    echo "  -i  : installs the oscml project as a python package"
    echo "  -d  : downloads project data from Vienna (will ask for your Vienna login details)"
    echo
    echo
    echo "==============================================================================================================="
    echo
fi
read -n 1 -s -r -p "Press any key to continue"
