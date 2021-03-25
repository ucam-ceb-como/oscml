import pkg_resources
import os

_RES_DIR = pkg_resources.resource_filename(__name__,os.path.join('..','resources'))
_RES_LOG_CONFIG_FILE = os.path.join(_RES_DIR,'logging.yaml')

DEFAULTS_ = {
"numerical_settings":{
	"seed": 1,
    "cudnn_deterministic": True,
    "cudnn_benchmark": False
},
"training":{
    "metric": "mse",
	"cross_validation": 0,
	"nested_cross_validation": 0,
	"direction": "minimize",
	"trials": 10,
    "jobs": 1,
    "timeout": None,
	"study_name": "oscml_study",
	"storage": None,
	"storage_timeout": 1000,
	"load_if_exists": True,
},
"logging_settings":{
	"log_main_dir": "./logs",
	"log_sub_dir_prefix": "job_",
	"log_config_file": _RES_LOG_CONFIG_FILE,
	"log_file_name": "oscml.log"
},
"post_processing":{
    "contour_plot": False,
	"contour_plot_alt_dir": None,
    "best_trial_retraining": False,
    "z_transform_inverse_prediction": False,
    "regression_plot": False,
    "transfer": False
},
"transfer_learning": {
  "fine_tune": False,
  "freeze_and_train": False
}
}