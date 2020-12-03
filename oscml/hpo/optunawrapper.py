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
    #pruner = optuna.pruners.HyperbandPruner(min_resource=1, max_resource=9, reduction_factor=3)
    pruner = None
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

def start_hpo(args, objective, log_dir, config, total_number_trials):

    #optuna.logging.enable_default_handler()
    #optuna.logging.enable_propagation()  # Propagate logs to the root logger.
    #optuna.logging.disable_default_handler()
    #optuna.logging.set_verbosity(optuna.logging.DEBUG)

    try:
        seed = config['numerical_settings'].get('seed')
        study_name = config['training'].get('study_name',args.study_name)
        direction = config['training'].get('direction')
        storage_url = config['training'].get('storage',args.storage)
        storage_timeout = config['training'].get('storage_timeout',5)
        load_if_exists = config['training'].get('load_if_exists',args.load_if_exists)
        n_trials = config['training'].get('n_trials',args.trials)
        n_jobs = config['training'].get('n_jobs',args.jobs)

        if storage_url:
            storage = optuna.storages.RDBStorage(
            url=storage_url,
            engine_kwargs={"connect_args": {"timeout": storage_timeout}},
            )
        else:
            storage = None

        study = create_study(direction=direction, seed=seed, storage=storage, study_name=study_name, load_if_exists=load_if_exists)
        decorator = create_objective_decorator(objective, total_number_trials)
        logging.info('starting HPO')
        study.optimize(decorator, n_trials=n_trials, n_jobs=n_jobs, timeout=args.timeout,
                catch = (RuntimeError, ValueError, TypeError), callbacks=[callback_on_trial_finished],
                gc_after_trial=True)
        logging.info('finished HPO')
        path = log_dir + '/hpo_result.csv'
        log_and_save(study, path)
        best_value = study.best_trial.value

        log_dir_best_trial_retrain = log_dir + '/best_trial_retrain'
        objective(study.best_trial, log_dir=log_dir_best_trial_retrain, best_trial_retrain=True)

        return best_value

    except BaseException as exc:
        print(exc)
        logging.exception('finished with exception', exc_info=True)
        raise exc
    else:
        logging.info('finished successfully')

def log_best_trial(trial):
    logging.info('best trial number=%s', trial.number)
    logging.info('best trial value=%s', trial.value)
    logging.info('best trial params=%s', trial.params)

def log_and_save(study, path):

    logging.info('Saving HPO results to %s', path)

    df = study.trials_dataframe()
    df.to_csv(path)

    logging.info('final study statistics: number of trials=%s', get_statistics(study))

    log_best_trial(study.best_trial)

def check_for_existing_study(storage, study_name):
    n_previous_trials = 0
    try:
        if storage:
            study_found = False
            summary = optuna.study.get_all_study_summaries(storage=storage)
            for existing_study in summary:
                if existing_study.study_name == study_name:
                    study_found = True
                    n_previous_trials = existing_study.n_trials
                    logging.info('found a study with name=%s and %s trials', storage, n_previous_trials)
                    if existing_study.best_trial:
                        log_best_trial(existing_study.best_trial)
                    else:
                        logging.info('no best trial so far')

        if not study_found:
            logging.info('there is no study with name=%s so far', storage)

    except:
        logging.info('exception - there is no study with name=%s so far', storage)
    return n_previous_trials
