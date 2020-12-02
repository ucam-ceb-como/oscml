#!/bin/bash
echo "This is a wrapper script for Slurm submission of the"
echo "ML models on CSD3."
echo

# Store command-line arguments for passing to Slurm script
args=$@


function usage {
    echo "Usage:"
    echo "  -c               : config file path                            "
    echo "                                                                 "
    echo "  -t  (hh:mm:ss)   : Estimated total wall-time                   "
    echo "                     Warning: Underestimate the run-time and your"
    echo "                     job will be killed pre-maturely...          "
    echo "                                                                 "
    echo "  -n               : job name (optional)                         "
    echo "  -h               : Display this help and exit                  "
    exit -1
}

JOBNAME="ML_run"
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
    -c) CONFIG=$2; shift;;
    -t) WALLT=$2; shift;;
    -n) JOBNAME=$2; shift;;
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


echo "The tests will be run on a single node of the skylake partition using 20 cores."
echo

usremailadr=$(git config user.email)

echo "Notification emails will be sent to: $usremailadr"
echo "(NB Edit your git config in order to change this.)"
echo

ml_exec=oscml/hpo/train.py

echo "Submitting job to Slurm..."

STDOUT1=$JOBNAME"_"slurm.%u.%j.%N.stdout.txt
STDERR1=$JOBNAME"_"slurm.%u.%j.%N.errout.txt

sbatch --mail-user=$usremailadr --time=$WALLT --output=$STDOUT1 --error=$STDERR1 ./SLURM_runhpo_csd3.sh $ml_exec --config $CONFIG --timeout $WALLTS

echo "Type \"squeue --jobs=<JOBID>\" or \"squeue -u $USER\" to watch it."
echo
echo "Done."

