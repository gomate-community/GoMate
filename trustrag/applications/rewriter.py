from trustrag.modules.rewriter.hyde_rewriter import HydeRewriter


class RewriterApp():
    """改写模块，用于理解和优化用户的原始问题。

    实现包括
    1. HyDE。用生成式大模型预回答，返回生成的假定答案。
    2. ...

    """

    def __init__(self, component_name=None):
        """Init required rewriter according to component name."""
        self.rewriter_list = ['hyde']
        assert component_name in self.rewriter_list
        if component_name == 'hyde':
            self.rewriter = HyDE_rewriter()

    def run(self, query, temperature=1e-10):
        """Run the required rewriter"""
        if query is None:
            raise ValueError('原始问题不能为空')
        return self.rewriter.run(query, temperature)
