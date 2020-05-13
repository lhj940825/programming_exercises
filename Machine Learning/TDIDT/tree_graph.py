import os

from graphviz import Digraph
import uuid
import json

dot = Digraph(comment='Decision Tree')


def generate_graph(p_index, tag, tree, depth):
    index = str(tree['index'])
    name = tree['name']
    num = tree['value']
    left = tree['left']
    right = tree['right']
    text = "index=" + index + "\nname=" + name + "\nvalue=" + str(num)

    dot.node(index, text)
    if p_index != -1:
        p_index = str(p_index)
        dot.edge(p_index, index, tag)

    if depth < 3:
        depth += 1
        sub_graph(dot, index, "<", left, depth)
        sub_graph(dot, index, ">=", right, depth)


def sub_graph(dot, index, tag, subtree, depth):
    if subtree == 1 or subtree == 0:
        text = "class=" + str(int(subtree))
        key = str(uuid.uuid4())
        dot.node(key, text)
        dot.edge(index, key, tag)
    else:
        generate_graph(index, tag, subtree, depth)


if __name__ == '__main__':
    output_folder = 'missing_value_output'
    json_filename = os.path.join(output_folder, "median.json")
    with open(json_filename) as f:
        lines = f.readlines()
        text = ''
        for l in lines:
            text += l.replace("\n", "")
        dict = json.loads(text)
        generate_graph(-1, '<', dict, 0)
        dot.format = 'png'
        dot.view('test_graph.dot')
