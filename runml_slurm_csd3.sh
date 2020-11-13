#!/bin/bash
echo "This is a wrapper script for Slurm submission of the"
echo "ML models on CSD3."
echo

# Store command-line arguments for passing to Slurm script
args=$@


function usage {
    echo "Usage:"
    echo "  -m  (BILSTM | SVR | RF | GNN)     : ML model to train"
    echo "  -d  (HOPV15 | CEP25000)           : Dataset"
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

while [[ $# > 0 ]]
do
key="$1"
case $key in
    -h)
     usage;;
    -m) MODEL=$2; shift;;
    -d) DATASET=$2; shift;;
    -t) WALLT=$2; shift;;
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

sbatch --mail-user=$usremailadr --job-name=$MODEL --time=$WALLT ./SLURM_runhpo_csd3.sh $ml_exec --dataset $DATASET --timeout $WALLTS --jobs -1


echo "Type \"squeue --jobs=<JOBID>\" or \"squeue -u $USER\" to watch it."
echo
echo "Done."

