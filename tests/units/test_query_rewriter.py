import pytest
from gomate.applications import RewriterApp
# import os

def test_rewriter():
    component_name = 'hyde'
    model = RewriterApp(component_name = component_name)
    query = ['gomate是哪天发布的？','gomate是做什么的？']
    answer = model.run(query)
    assert answer is not None

if __name__ == '__main__':
    test_rewriter()