#!/usr/bin/env python3


class Reader:
    def __init__(self, paths):
        if type(paths) is str:
            self.paths = [paths]
        else:
            self.paths = list(paths)

    def AddPaths(self, paths):
        self.paths = self.paths + [paths] if type(paths) is str else \
            self.paths + paths

    def GetData(self, limit, containsSkip=None):
        lines = []

        for path in self.paths:
            with open(path, 'r') as f:
                for line in f.readlines():
                    lines.append(line)
                    if len(lines) == limit:
                        yield lines
                        lines = []
        yield lines


reader = Reader(
    '/home/smazle/Git/Authorship-Verification/data/pan_2015/EN001/known01.txt')
reader.AddPaths(
    '/home/smazle/Git/Authorship-Verification/data/pan_2015/EN001/unknown.txt')

reader = reader.GetData(50)


while True:
    input()
    print(next(reader))
