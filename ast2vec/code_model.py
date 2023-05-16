from torch import Tensor
import ast2vec.tree
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
import ast 

from ast2vec import tree_grammar, python_ast_utils
from ast2vec.rtgae2 import RTGAE

class CodeEmbeddingModel(nn.Module):
    def __init__(self, device='cpu', from_pretrained=True, output_dim=256):
        super(CodeEmbeddingModel, self).__init__()
        self.device = device
        self.out_dim = output_dim
        self.model = RTGAE(python_ast_utils.grammar, dim = self.out_dim)
        self.model.to(self.device)
        self.grammar = tree_grammar.TreeGrammar(python_ast_utils._alphabet, 
                                                python_ast_utils._nonterminals, 
                                                python_ast_utils._start, 
                                                python_ast_utils._rules)

        self.parser = tree_grammar.TreeParser(self.grammar)

        if from_pretrained:
            self.model.load_state_dict(torch.load('ast2vec/ast2vec.pt'))

    def code2tree(self, code):
        tree = python_ast_utils.ast_to_tree(ast.parse(code, 'program.py'))
        self.parser.parse_tree(tree)
        return tree
    
    def forward(self, x):
        tree = self.code2tree(x)
        x = self.model.encode(tree)
        return x