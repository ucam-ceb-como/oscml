
import argparse

import oscml.utils.util
import oscml.data.dataset_cep

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='.')
    parser.add_argument('--dst', type=str, default='.')
    parser.add_argument('--task', type=int, choices=range(1,3), required=True)
    args = parser.parse_args()

    oscml.utils.util.init_logging('.', '.')

    if args.task==1:
        # python ./oscml/data/start.py --task 1 --src ./data/raw/CEPDB.csv --dst ./data/processed/CEPDB_valid_SMILES_tmp.csv
        oscml.data.dataset_cep.store_CEP_with_valid_SMILES(args.src, args.dst)