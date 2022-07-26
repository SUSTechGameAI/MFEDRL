"""
  @Time : 2021/9/16 20:40 
  @Author : Ziqi Wang
  @File : train.py
"""

import argparse
from src.designer import train_designer
from src.gan import train_gan
from src.repairer import cnet


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_g = subparsers.add_parser('generator', help='Train GAN generator')
    parser_d = subparsers.add_parser('designer')
    parser_cnet = subparsers.add_parser('cnet')

    train_gan.set_parser(parser_g)
    parser_g.set_defaults(entry=train_gan.train_gan)

    train_designer.set_parser(parser_d)
    parser_d.set_defaults(entry=train_designer.train_designer)

    cnet.set_parser(parser_cnet)
    parser_cnet.set_defaults(entry=cnet.train_cnet)

    cfgs = parser.parse_args()
    entry = cfgs.entry
    entry(cfgs)
