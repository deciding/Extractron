import argparse
import os
from time import sleep

import infolog
import tensorflow as tf
from hparams import hparams
from infolog import log
from extractron.train import extractron_train

#python train.py --dataset xyz_mel --eval_dataset xyz_mel_test --name extractron_mel_bi

def save_seq(file, sequence, input_path):
    '''Save Extractron training state to disk. (To skip for future runs)
    '''
    sequence = [str(int(s)) for s in sequence] + [input_path]
    with open(file, 'w') as f:
        f.write('|'.join(sequence))


def read_seq(file):
    '''Load Extractron training state from disk. (To skip if not first run)
    '''
    if os.path.isfile(file):
        with open(file, 'r') as f:
            sequence = f.read().split('|')

        return [bool(int(s)) for s in sequence[:-1]], sequence[-1]
    else:
        return [0, 0, 0], ''


def prepare_run(args):
    modified_hp = hparams.parse(args.hparams)
    if args.hparams_json:
        import json
        with open(args.hparams_json) as hp_json_file:
            hp_json=json.dumps(json.load(hp_json_file)['hparams'])
            modified_hp=modified_hp.parse_json(hp_json)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = str(args.tf_log_level)
    run_name = args.name or args.model
    log_dir = os.path.join(args.base_dir, 'logs-{}'.format(run_name))
    os.makedirs(log_dir, exist_ok=True)
    infolog.init(os.path.join(log_dir, 'Terminal_train_log'),
                 run_name, args.slack_url)
    return log_dir, modified_hp


def train(args, log_dir, hparams):
    state_file = os.path.join(log_dir, 'state_log')
    # Get training states
    extrac_state, input_path = read_seq(state_file)

    if not extrac_state:
        log('\n#############################################################\n')
        log('Extractron Train\n')
        log('###########################################################\n')
        checkpoint = extractron_train(args, log_dir, hparams)
        tf.reset_default_graph()
        # Sleep 1/2 second to let previous graph close and avoid error messages while synthesis
        sleep(0.5)
        if checkpoint is None:
            raise('Error occured while training Extractron, Exiting!')
        extrac_state = 1
        save_seq(state_file, extrac_state, input_path)
    else:
        checkpoint = os.path.join(log_dir, 'extrac_pretrained/')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', default='')
    parser.add_argument('--dataset', default='')
    parser.add_argument('--eval_dataset', default='')
    parser.add_argument('--hparams', default='',
                        help='Hyperparameter overrides as a comma-separated list of name=value pairs')
    parser.add_argument('--hparams_json', default='',
                        help='Hyperparameter in json format')
    parser.add_argument('--name', default='extractron', help='Name of logging directory.')
    parser.add_argument('--model', default='Extractron')
    parser.add_argument('--restore', type=bool, default=True,
                        help='Set this to False to do a fresh training')
    parser.add_argument('--summary_interval', type=int, default=200,
                        help='Steps between running summary ops')
    parser.add_argument('--gc_interval', type=int, default=100,
                        help='Steps between garbage collecting')
    parser.add_argument('--checkpoint_interval', type=int, default=1000,
                        help='Steps between writing checkpoints')
    parser.add_argument('--super_checkpoint_interval', type=int, default=10000,
                        help='Steps between writing checkpoints')
    parser.add_argument('--eval_interval', type=int, default=200,
                        help='Steps between eval on test data')
    parser.add_argument('--test_steps', type=int, default=5,
                        help='Steps between eval on test data')
    parser.add_argument('--extractron_train_steps', type=int,
                        default=2000000, help='total number of extractron training steps')
    parser.add_argument('--tf_log_level', type=int,
                        default=1, help='Tensorflow C++ log level.')
    parser.add_argument('--slack_url', default=None,
                        help='slack webhook notification destination link')
    args = parser.parse_args()

    accepted_models = ['Extractron']

    if args.model not in accepted_models:
        raise ValueError(
            'please enter a valid model to train: {}'.format(accepted_models))

    log_dir, hparams = prepare_run(args)

    if args.model == 'Extractron':
        extractron_train(args, log_dir, hparams)
    elif args.model == 'Extractron-2':
        train(args, log_dir, hparams)
    else:
        raise ValueError('Model provided {} unknown! {}'.format(
            args.model, accepted_models))

if __name__ == '__main__':
    main()
