#!/bin/bash
# D. Nurkowski (danieln@cmclinnovations.com)


AUTHOR="Daniel Nurkowski <danieln@cmclinnovations.com>"
SPATH="$( cd  "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/"

DATA_LOCAL="./data/raw"
DATA_REMOTE="vienna.cheng.cam.ac.uk:/home/userspace/CoMoCommon/Ongoing/Projects/c4e-jps-OSC/Data/Raw/*.*"
# Read project conda environment name from the environment.yml file
OLDIFS=$IFS
IFS=': '
read tag CONDA_ENV_NAME < "./environment.yml"
IFS=$OLDIFS

echo

function check_conda {
    echo "1. Verifying conda installation."
    echo "-------------------------------------------------------------"
    conda_version=$(conda --version)
    if [ $? -eq 0 ]; then
        echo
        echo "INFO: Found "$conda_version
    else
        echo "ERROR: Couldn't find conda installation. On Windows, you must run this script from Anaconda Prompt for conda to be correctly located. Aborting installation."
        read -n 1 -s -r -p "Press any key to continue"
        exit -1
    fi
    echo
    echo
}

function recreate_conda_env {
    echo "2. Creating / updating conda project '"$CONDA_ENV_NAME"' environment."
    echo "-------------------------------------------------------------"
    # This will create a new environment if one doesn't exist or update the existing one by installing packages listed in the environment.yml file and removing those packaged that are not listed. The environment name will be as specified in the environment.yml file.
    echo
    conda env update -f environment.yml --prune
    if [ $? -ne 0 ]; then
        echo
        echo "ERROR: Couldn't create / update conda '"$CONDA_ENV_NAME"' environment. Aborting installation."
        read -n 1 -s -r -p "Press any key to continue"
        exit -1
    fi
    echo
    echo
}

function install_ZhouLiML_package {
    echo "3. Installing the OSCML project as a python package..."
    echo "-------------------------------------------------------------"
    echo
    source activate $CONDA_ENV_NAME

    if [ $? -eq 0 ]; then
        pip install -e .
        if [ $? -ne 0 ]; then
			echo "ERROR: Couldn't install the OSCML project. Please check the pip log."
			exit -1
        fi
    else
        echo "Couldn't activate conda '"$CONDA_ENV_NAME"' environment. Aborting installation."
        read -n 1 -s -r -p "Press any key to continue"
        exit -1
    fi
    echo
    echo
}

function get_data_from_server {
    echo "4. Downloading required project data from the server..."
    echo "-------------------------------------------------------------"
    echo
    if [ -d $DATA_LOCAL ]
    then
        echo "INFO: Directory" $DATA_LOCAL "already exists. Remove it if you wish to re-download data from the server."
    else
        mkdir -p $DATA_LOCAL
        echo -n "Please provide your Vienna user-name: "
        read USERNM
        scp $USERNM"@"$DATA_REMOTE $DATA_LOCAL"/."
    fi
    echo
    echo
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
    echo "  -d  : downloads project data from Vienna, (will ask for your Vienna login details)"
fi
echo
echo "==============================================================================================================="
echo
read -n 1 -s -r -p "Press any key to continue"
