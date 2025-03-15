import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='OTRec', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='baby', help='name of datasets')
    parser.add_argument('--gpu_id', '-g', type=int, default='7', help='gpu id')
    parser.add_argument('--embedding_size', '-e', type=int, default='64', help='embedding_size')
    
    args, _ = parser.parse_known_args()
    config_dict = {
        #'gpu_id': 0,
        'gpu_id': args.gpu_id,
        'embedding_size':args.embedding_size,
    }



    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


