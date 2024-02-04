from gomate.modules import Rewriter
# import os

def test_rewriter():
    component_name = 'hyde'
    model = Rewriter(component_name = component_name)
    query = ['gomate是哪天发布的？','gomate是做什么的？']
    answer = model(query)
    assert answer is not None

if __name__ == '__main__':
    test_rewriter()