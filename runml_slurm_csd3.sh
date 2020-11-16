#!/bin/bash
echo "This is a wrapper script for Slurm submission of the"
echo "ML models on CSD3."
echo

# Store command-line arguments for passing to Slurm script
args=$@


function usage {
    echo "Usage:"
    echo "  -m  (BILSTM | SVR | RF | GNN |    : ML model to train."
    echo "       DISTR_BILSTM)"
    echo "  -d  (HOPV15 | CEP25000)           : Dataset."
    echo "  -s  (None | sqlite:///filename.db): Storage for distributed hpo."
    echo "  -n  (None | <your name>)          : Study name for distributed hpo."
    echo "  -l  (False | True)                : Load study from the storage"
    echo "                                      for distributed hpo."
    echo "  -t  (hh:mm:ss)                    : Estimated total wall-time."
    echo "                                      Warning: Underestimate the run-time and your"
    echo "                                      job will be killed pre-maturely..."
    echo "  -h                                : Display this help and exit."
    exit -1
}


# Scan command-line arguments
if [[ $# = 0 ]]
then
   usage
fi


STORAGE=None
STUDY_NAME=None
LOAD_IF_EXISTS=False

while [[ $# > 0 ]]
do
key="$1"
case $key in
    -h)
     usage;;
    -m) MODEL=$2; shift;;
    -d) DATASET=$2; shift;;
    -t) WALLT=$2; shift;;
    -s) STORAGE=$2; shift;;
    -n) STUDY_NAME=$2; shift;;
    -l) LOAD_IF_EXISTS=$2; shift;;
    *)
    usage
    # otherwise do nothing
    ;;
esac
shift # past argument
done

# Split time into hh, mm and ss
hms=(${WALLT//:/ })

# Calculate total time in ss, or the timeout parameter
WALLTS=$((3600 * ${hms[0]} + 60 * ${hms[1]} + ${hms[2]}))
# Make timeout 1% smaller than the total slurm time
WALLTS=$(($WALLTS * 99/100))

# Check inputs
# -d
if [ "$DATASET" != "HOPV15" ] && [ "$DATASET" != "CEP25000" ]
then
    echo "Unknown dataset: "$DATASET
    exit -1
fi

# -m
if [ "$MODEL" = "SVR" ]
then
    ml_exec=oscml/hpo/start_svr_with_hpo.py

elif [ "$MODEL" = "GNN" ]
then
    ml_exec=oscml/hpo/start_gnn_with_hpo.py

elif [ "$MODEL" = "BILSTM" ]
then
     ml_exec=oscml/hpo/start_bilstm_with_hpo.py
elif [ "$MODEL" = "RF" ]
then
     ml_exec=oscml/hpo/start_rf_with_hpo.py
elif [ "$MODEL" = "DISTR_BILSTM" ]
then
     ml_exec=oscml/hpo/start_bilstm_with_hpo_and_storage.py
else
    echo "Unknown model choice: "$MODEL
    exit -1
fi


echo "The tests will be run on a single node of the skylake partition using 20 cores."
echo

usremailadr=$(git config user.email)

echo "Notification emails will be sent to: $usremailadr"
echo "(NB Edit your git config in order to change this.)"
echo

#usrworkdir=$(pwd)

echo "Submitting job to Slurm..."

sbatch --mail-user=$usremailadr --job-name=$MODEL --time=$WALLT ./SLURM_runhpo_csd3.sh $ml_exec --dataset $DATASET --timeout $WALLTS --jobs -1 --storage $STORAGE --study_name $STUDY_NAME --load_if_exists $LOAD_IF_EXISTS


echo "The tests will be run on a single node of the skylake partition using 20 cores."
echo

usremailadr=$(git config user.email)

echo "Notification emails will be sent to: $usremailadr"
echo "(NB Edit your git config in order to change this.)"
echo

#usrworkdir=$(pwd)

echo "Submitting job to Slurm..."




echo "Type \"squeue --jobs=<JOBID>\" or \"squeue -u $USER\" to watch it."
echo
echo "Done."

