import argparse
import logging
import os

import oscml.utils.util
import oscml.data.dataset_cep

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='.')
    parser.add_argument('--dst', type=str, default='.')
    parser.add_argument('--task', type=int, choices=range(1,4), required=True)
    args = parser.parse_args()

    oscml.utils.util.init_logging('.', '.')

    try:
        logging.info('starting task=' + str(vars(args)))

        dir_name = os.path.dirname(args.dst)
        try:
            os.makedirs(dir_name, exist_ok=True)
        except FileExistsError:
            logging.info('destination dir already exists, dir=' + dir_name)

        if args.task==1:
            # python ./oscml/data/start.py --task 1 --src ./data/raw/CEPDB.csv --dst ./data/processed/CEPDB_valid_SMILES.csv
            oscml.data.dataset_cep.store_CEP_with_valid_SMILES(args.src, args.dst, numbersamples=1000)
        elif args.task==2:
            # python ./oscml/data/start.py --task 2 --src ./data/processed/CEPDB_valid_SMILES.csv --dst ./data/processed/CEPDB_25000_stratified.csv
            oscml.data.dataset_cep.store_CEP_cleaned_and_stratified(
                    args.src, args.dst, number_samples=25000, threshold=0.0001)
        """    
        elif args.task==3:
            # python ./oscml/data/start.py --task 2 --src ./data/processed/CEPDB_valid_SMILES.csv --dst ./data/processed/CEPDB_25000_stratified.csv
            oscml.data.dataset_cep.store_CEP_cleaned_and_stratified(
                    args.src, args.dst, number_samples=25000, threshold=0.0001)
        """
        
    except Exception as exc:
        logging.exception('task failed', exc_info=True)
        raise exc
    else:
        logging.info('tasked finished successfully')