import torch, argparse
from allennlp.nn import util

import re

cuda_device = -1

def convert(weights_in_file, weights_out_file):
    weights_in = torch.load(weights_in_file, map_location=util.device_mapping(cuda_device))

    weights_out = {}
    for name, w in weights_in.items():
        if 'encoder' in name:
            name_new = re.sub('(layer_\\d+)', '\\1.cell', name)
            # print("%s->%s" % (k, k_new))
        else:
            name_new = name

        weights_out[name_new] = w

    torch.save(weights_out, weights_out_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upgrade a weight file to AllenNLP 1.1")
    parser.add_argument('--input', type=str)
    parser.add_argument('--out', type=str)

    args = parser.parse_args()
    convert(args.input, args.out)
 
