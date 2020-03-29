from models import *
from params import VPATH
from base import Video

from os.path import join
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('file', type=str, default='random_video.avi')
parser.add_argument('-s', '--save', action='store_true')
parser.add_argument('-e', '--epochs', type=int, default=20)
parser.add_argument('-k', '--skip', type=int, default=1)
parser.add_argument('-l', '--load', action='store_true')
parser.add_argument('-r', '--rect', default=(100,100))

if __name__ == '__main__':
    args = parser.parse_args()

    source = join(VPATH, args.file)
    video = Video(source, skip=args.skip)

    load = args.load
    if not load:
        net = PerseptronModel(video)
    else:
        net = PerseptronModel()

    net.fit(epochs=20)
    if args.save:
        net.save()
