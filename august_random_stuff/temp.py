# -*- coding: utf-8 -*-
import json
from collections import defaultdict

jason = json.loads(open('./graph.json', 'r').read())
outputNode = jason['config']['output_layers'][0][0]
jason = jason['config']['layers']

jason = {x['name']: x for x in jason}

# print json.dumps(jason, indent=4, sort_keys=True)

newJason = {key.replace('_', ''): {'name': val['name'].replace('_', ''),
                                   'parents':
                                   [x[0][0] for x in val['inbound_nodes']],
                                   'config': val['config'],
                                   'children': []}
            for key, val in jason.iteritems()}


tikzFile = open('tikz', 'w')

tikz = ['\\begin{tikzpicture}[sibling distance=10em,',
        'every node/.style = {shape=rectangle, rounded corners,',
        'draw, align=center,',
        'top color=white, bottom color=blue!20}]\n']

tikzFile.writelines(tikz)

lines = []


prev = []


def GetChildren(node):
    parents = node['parents']
    map(lambda x: newJason[x]['children'].append(node['name']), parents)
    [GetChildren(newJason[x]) for x in parents]


GetChildren(newJason[outputNode.replace('_', '')])
print(newJason[newJason.keys()[0]])


def GenerateTree(node, count):
    for i in list(set(node['children'])):
        child = newJason[i]
        lines.append('child { node {' + child['name'] + '} ')
        if child['children'] == []:
            for i in range(count):
                lines[-1] += '}'
            lines[-1] += '\n'
            count = 0
        else:
            lines[-1] += '\n'

        # import pdb; pdb.set_trace()
        GenerateTree(child, count + 1)


inputNodes = list(filter(lambda x: len(
    newJason[x]['parents']) == 0, newJason.keys()))
for idx, i in enumerate(inputNodes):
    inp = newJason[i]
    if idx > 0:
        tikzFile.write(
            '\\node (' + i + ') [right of ' +
            inputNodes[idx - 1] + '] {' + i + '}\n')
    else:
        tikzFile.write('\\node (' + i + ') {' + i + '}\n')
    GenerateTree(inp, 1)
    tikzFile.write(';\n')

tikzFile.write('\n\\end{tikzpicture}')
