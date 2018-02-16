import argparse
import sys
import warnings

import cv2

from examples.screen_demo import live_demo
from examples.static_images import samples
from src.judge import Judge
from src.paths import DataPath
from src.process import CollectProcess

try:
    from src.setup import setup_tensor
except ImportError:
    warnings.warn('Object_detection module not found')


def collect(mode, w=800, h=600, lim=None):

    print(w, h)

    c = CollectProcess('debug')

    if mode == 'video':
        c.collect_from_video()
    elif mode == 'screen':
        c.collect_from_screen(w, h)


def judge(clf):

    print(clf)
    j = Judge()

    if clf == 'tensorflow':
        dg, ci = setup.setup_tensor()
        j.evaluate_tensorflow(dg, ci)


def main():
    """
    Main function of the aiminator.py - captures the screen of the size 800x600 in
    the top-left corner of the screen and collects 100 samples to evaluate using judge function.
    """
    collect('screen', lim=100)
    judge('tensorflow')


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('--demo', help='Demo the process module. Available modes: live, samples')
    parser.add_argument('--collect', help='Collect using method: screen, video')

    parser.add_argument('-width', help='Screen width for collect process', type=int, default=800)
    parser.add_argument('-height', help='Screen height for collect process', type=int, default=600)

    parser.add_argument('--judge', help='Judge the collected data using: tensorflow, human')

    args = parser.parse_args()

    try:
        if len(sys.argv) > 1:
            if args.demo == 'live':
                live_demo()
            elif args.demo == 'samples':
                samples('images/examples/')
            if args.collect:
                collect(args.collect, w=args.width, h=args.height)
            if args.judge:
                judge(args.judge)
        else:
            main()

    except argparse.ArgumentError:
        print('Invalid arguments.')
