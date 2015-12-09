import os
import os.path as osp
import lmdb
import sys
from argparse import ArgumentParser

if 'python' not in sys.path:
    sys.path.insert(0, 'python')
import caffe


def main(args):
    if not osp.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    datum = caffe.proto.caffe_pb2.Datum()
    with lmdb.open(args.input_db) as env:
        with env.begin() as txn:
            cursor = txn.cursor()
            for i, (key, value) in enumerate(cursor):
                if args.max_num is not None and i >= args.max_num: break
                name = osp.splitext(osp.basename(key))[0]
                datum.ParseFromString(value)
                file_path = osp.join(args.output_dir,
                                     '{:010d}_{}.jpg'.format(i, name))
                with open(file_path, 'wb') as f:
                    f.write(datum.data)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('input_db')
    parser.add_argument('output_dir')
    parser.add_argument('--max_num', type=int)
    args = parser.parse_args()
    main(args)