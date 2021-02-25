#!/bin/bash
# D. Nurkowski (danieln@cmclinnovations.com)

AUTHOR="Daniel Nurkowski <danieln@cmclinnovations.com>"
SPATH="$( cd  "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/"

DATA_LOCAL="./data/processed"
DATA_REMOTE="<REPLACE_ME>"
echo

function check_conda {
    echo "Verifying conda installation."
    echo "-------------------------------------------------------------"
    conda_version=$(conda --version)
    if [ $? -eq 0 ]; then
        echo
        echo "INFO: Found "$conda_version
    else
        echo "ERROR: Could not find conda installation. On Windows, you must run this script from Anaconda Prompt for conda to be correctly located. Aborting installation."
        read -n 1 -s -r -p "Press any key to continue"
        exit -1
    fi
    echo
    echo
}

function recreate_conda_env {
    echo "Creating / updating conda environment and installing the oscml package."
    echo "-------------------------------------------------------------"
    # This will recreate conda environment
	echo -n "Provide conda environment name to be created for this project: "
    read CONDA_ENV_NAME
	# Remove the environment (if one already exists)
	conda remove --name $CONDA_ENV_NAME --all
	# Update the environment name in the yml file
    sed -i '1s/.*/name: '$CONDA_ENV_NAME'/' environment.yml
    # Create the new environment
    conda env create -f environment.yml
    echo
    echo
}

function get_data_from_server {
    echo "Downloading required project data..."
    echo "-------------------------------------------------------------"
    echo
    curl $DATA_REMOTE -o $DATA_LOCAL"/data.zip"
    unzip $DATA_LOCAL"/data.zip" -d $DATA_LOCAL
    rm -f $DATA_LOCAL"/data.zip"
    echo
    echo
}

if [ "$1" == "-i" ]
then
    check_conda
    recreate_conda_env
elif [ "$1" == "-d" ]
then
    get_data_from_server
elif [ "$1" == "-a" ]
then
    check_conda
    recreate_conda_env
    get_data_from_server
else
    echo "==============================================================================================================="
    echo "OSCML project installation script."
    echo
    echo "Please run the script with one of the following flags set:"
    echo "--------------------------------------------------------------------------------------------------------"
    echo "  -i  : creates conda environment for this project and installs the oscml package in it including all"
    echo "        the necessary dependencies"
    echo "  -d  : downloads project data from the remote location"
    echo "  -a  : performs all the above steps in one go"

fi
echo
echo "==============================================================================================================="
echo
read -n 1 -s -r -p "Press any key to continue"
