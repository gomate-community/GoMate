from trustrag.modules.rewriter.openai_rewrite import OpenaiRewriter,OpenaiRewriterConfig
if __name__ == '__main__':

    rewriter_config = OpenaiRewriterConfig(
            api_url="https://aicloud.oneainexus.cn:30013/inference/aicloud-yanqiang/gomatellm/"
        )
    openai_rewriter = OpenaiRewriter(rewriter_config)

    query="在“一带一路”国际合作高峰论坛上，习近平讲了什么？"
    query="习近平关于改革开放有什么最新的论述？"

    result=openai_rewriter.rewrite(query)
    print(result)