from docopt import docopt, DocoptExit
from oscml.jobhandling import JobHandler


__doc__ = """oscml_run
Usage:
    oscml <configFile> [options]

    options:
        --trials=TRIALS                     No. of hpo trials to run
        --jobs=JOBS                         No. of parallel hpo jobs
        --epochs=EPOCHS                     No. of epochs to use during training (if applicable)
        --batch_size=BATCHSIZE              Batch size to use during training (if applicable)
        --patience=PATIENCE                 Early stopping patience parameter to use during training (if applicable)
        --min_delta=MINDELTA                Early stopping min delta parameter to use during training (if applicable)
        --cross_validation=CROSSVAL         Use inner cross validation: False | no. of folds
        --study_name=STUDYNAME              Name of the study
        --storage=STORAGE                   Use sqlite storage for this run:
                                            True | False | alternative storage name
        --timeout=TIMEOUT                   Stop study after the given number of second(s). If this argument is not set,
                                            the study is executed without time limitation.
        --storage_timeout=STORTIMEOUT       Maximum time for the storage access operations
        --load_if_exists=LOADIFEXISTS       In case where a study already exists in the storage load it
        --log_config_file=LOGCONFIG         Logging config file
        --log_main_dir=LOGDIR               Main logging directory
        --log_sub_dir_prefix=LOGSUBDIRPREF  Log subfolder prefix
        --log_file_name=LOGBASENAME         Log file base name
        --seed=SEED                         Random seed to use
"""

def run():
    try:
        args = docopt(__doc__)
    except DocoptExit:
        raise DocoptExit('Error: oscml called with wrong arguments.')


    jobHandler = JobHandler(args)
    jobHandler.runJob()

if __name__ == '__main__':
    run()

