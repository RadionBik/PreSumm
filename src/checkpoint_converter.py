import argparse
import json
import pathlib

import torch

""" 
This converter is needed to reduce size of checkpoints, those are also pickled objects and require source classes
to be present when unpickled that makes them unportable. 
Converted checkpoint do not include optimizers and training cannot be continued
"""


def convert_model_to_portable(model_path):
    model_path = pathlib.Path(model_path)
    print(f'loading checkpoint from {model_path}')
    checkpoint = torch.load(model_path.as_posix())
    print('loaded source checkpoint')
    base_name = model_path.name.split('.')[0]
    target_dir = model_path.parent / base_name
    target_dir.mkdir(exist_ok=True)

    torch.save(checkpoint['model'], target_dir / 'model.bin')
    print('saved parameter dict')
    arg_dict = vars(checkpoint['opt'])
    with open(target_dir / 'config.json', 'w') as cf:
        json.dump(arg_dict, cf)
    print('saved config')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint',
                        help='path to a saved checkpoint file')

    args = parser.parse_args()
    convert_model_to_portable(args.checkpoint)


if __name__ == '__main__':
    main()

