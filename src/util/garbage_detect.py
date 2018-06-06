#!/usr/bin/python3
# -*- coding: utf-8 -*-


def is_garbage(text):
    length = len(text)
    spaces_length = len(list(filter(lambda x: x == ' ', text)))
    ratio = spaces_length / length

    return ratio < 0.05 or ratio > 0.95


if __name__ == '__main__':
    print(is_garbage('Hej med dig jeg hedder mor og jeg bor i et hus.'))
