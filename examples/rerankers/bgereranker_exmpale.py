#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: main.py
@time: 2024/06/06
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""

from trustrag.modules.reranker.bge_reranker import BgeReranker, BgeRerankerConfig

if __name__ == '__main__':
    reranker_config = BgeRerankerConfig(
        model_name_or_path="/data/users/searchgpt/pretrained_models/bge-reranker-large"
    )

    bge_reranker = BgeReranker(reranker_config)
    contexts = [
        "RCEP(Regional Comprehensive Economic Partnership)，即区域全面经济伙伴关系协定  　　RCEP是Regional Comprehensive Economic Partnership的缩写，即区域全面经济伙伴关系协定，RCEP由东盟十国发起，邀请中国、日本、韩国、澳大利亚、新西兰、印度共同参加(“10+6”)，通过削减关税及非关税壁垒，建立16国统一市场的自由贸易协定。  　　RCEP是东盟国家近年来首次提出，并以东盟为主导的区域经济一体化合作，是成员国间相互开放市场、实施区域经济一体化的组织形式。RCEP主要成员国计划包括与东盟已经签署自由贸易协定的国家，即中国、日本、韩国、澳大利亚、新西兰、印度。",
        "2万亿美元，约占全球GDP的30%，占全球贸易总额的近28%（根据2019年数据），RCEP是朝着全球贸易和投资规则的理想框架迈出的重要一步。其次，RCEP是区域内经贸规则的“整合器”。RCEP整合了东盟与中国、日本、韩国、澳大利亚、新西兰多个“10+1”自贸协定以及中、日、韩、澳、新西兰5国之间已有的多对自贸伙伴关系，尤其是在中日和韩日间建立了新的自贸伙伴关系。",
        "RCEP最早由东盟十国发起，邀请中国、日本、韩国、澳大利亚、新西兰、印度共同参加（“10+6”），旨在通过削减关税及非关税壁垒，建立16国统一市场的自由贸易协定。这意味着，如果达成，RCEP将会形成人口约35亿、GDP总和约为23万亿美元、占世界贸易总量约30%的贸易集团。这在给所有参与国家带来实质性贸易量增加的同时，也将会给各国企业在地区与国际市场扩大投资和增加市场份额带来莫大实惠。",
        "7亿，GDP达26万亿美元，出口总额达5.2万亿美元，均占全球总量约30%。RCEP自贸区的建成意味着全球约三分之一的经济体量将形成一体化大市场。RCEP囊括了东亚地区主要国家，将为区域和全球经济增长注入强劲动力。  　　RCEP是区域内经贸规则的“整合器”。RCEP整合了东盟与中国、日本、韩国、澳大利亚、新西兰多个“10+1”自贸协定以及中、日、韩、澳、新西兰5国之间已有的多对自贸伙伴关系，还在中日和日韩间建立了新的自贸伙伴关系。",
        "此外，成员国还在知识产权、电子商务、竞争、政府采购和中小企业等领域制订了高标准的自由贸易规则。 　　根据RCEP规定，协定签署后，RCEP各成员国将各自履行国内法律审批程序。协定生效需15个成员中至少9个成员批准，其中至少包括6个东盟成员国和中国、日本、韩国、澳大利亚和新西兰中至少3个国家。 　　新签署的协定吸引了全世界的注意力，具有以下重要意义：首先，RCEP是当今世界上最大的自由贸易协定。该协定覆盖22亿人口，约占世界人口的30%，国内生产总值（GDP）达26.",
        "Nov 15, 2020 ... RCEP成员国包括东盟10国与中国、日本、韩国、澳大利亚、新西兰。RCEP是全球最大的自贸协定，15个成员国总人口、经济体量、贸易总额均占全球总量约30% ..."]
    top_docs = bge_reranker.rerank(
        query="RCEP具体包括哪些国家",
        documents=contexts,
    )
    print(top_docs)
