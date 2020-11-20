import logging
import os

import optuna
import optuna.samplers


os.environ["SLURM_JOB_NAME"]="bash"

def create_objective_decorator(objective, n_trials):
    def decorator(trial):
        try:
            logging.info('starting trial %s / %s', trial.number, n_trials)
            value = objective(trial)
            logging.info('finished trial %s / %s', trial.number, n_trials)
            return value
        except optuna.exceptions.TrialPruned as exc:
            logging.info('pruned trial, trial number=%s', trial.number)
            raise exc
        except Exception as exc:
            message = 'failed trial, trial number=' + str(trial.number)
            logging.exception(message, exc_info=True)
            raise exc

    return decorator


def create_study(direction, seed, **kwargs):
    # pruner = optuna.pruners.MedianPruner()
    # pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=30, interval_steps=10)
    # pruner = optuna.pruners.PercentilePruner(25.0, n_startup_trials=5, n_warmup_steps=30, interval_steps=10) #keep top 25%
    # #pruner = ThresholdPruner(upper=1.0)
    pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=9, reduction_factor=3)
    #pruner = None
    sampler = optuna.samplers.TPESampler(consider_prior=True, n_startup_trials=10, seed=seed)
    study = optuna.create_study(direction=direction, pruner=pruner, sampler=sampler, **kwargs)
    return study

def get_statistics(study):
    running_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING]
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    return {'all': len(study.trials), 'running': len(running_trials), 'completed': len(completed_trials),
            'pruned': len(pruned_trials), 'failed': len(failed_trials)}

def callback_on_trial_finished(study, trial):
    statistics = get_statistics(study)
    logging.info('current study statistics: number of trials=%s', statistics)
    if statistics['failed'] >= 50:
        logging.error('THE MAXIMUM NUMBER OF FAILED TRIALS HAS BEEN REACHED, AND THE STUDY WILL STOP NOW.')
        study.stop()

def start_hpo(args, objective, log_dir):

    #optuna.logging.enable_default_handler()
    #optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    #optuna.logging.disable_default_handler()
    #optuna.logging.set_verbosity(optuna.logging.DEBUG)

    try:

        study = create_study(direction=args.direction, seed=args.seed, storage=args.storage, study_name=args.study_name, load_if_exists=args.load_if_exists)
        decorator = create_objective_decorator(objective, args.trials)
        logging.info('starting HPO')
        study.optimize(decorator, n_trials=args.trials, n_jobs=args.jobs, timeout=args.timeout,
                catch = (RuntimeError, ValueError, TypeError), callbacks=[callback_on_trial_finished],
                gc_after_trial=True)
        logging.info('finished HPO')
        path = log_dir + '/hpo_result.csv'
        log_and_save(study, path)
        best_value = study.best_trial.value

        return best_value

    except BaseException as exc:
        print(exc)
        logging.exception('finished with exception', exc_info=True)
        raise exc
    else:
        logging.info('finished successfully')


def log_and_save(study, path):

    logging.info('Saving HPO results to %s', path)

    df = study.trials_dataframe()
    df.to_csv(path)

    logging.info('final study statistics: number of trials=%s', get_statistics(study))

    trial = study.best_trial
    logging.info('best trial number=%s', trial.number)
    logging.info('best trial value=%s', trial.value)
    logging.info('best trial params=%s', trial.params)
