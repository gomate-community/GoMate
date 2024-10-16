#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author:quincy qiang
@license: Apache Licence
@file: bodys.py
@time: 2024/06/13
@contact: yanqiangmiffy@gamil.com
@software: PyCharm
@description: coding..
"""
from typing import List

from pydantic import BaseModel, Field


class CitationBody(BaseModel):
    """
    # 入参模型定义
    """
    question: str = Field(
        "请介绍下巨齿鲨2电影",
        description="答案"
    )
    response: str = Field(
        "巨齿鲨2是一部科幻冒险电影，由本·维特利执导，杰森·斯坦森、吴京、蔡书雅和克利夫·柯蒂斯主演。电影讲述了海洋霸主巨齿鲨，今夏再掀狂澜！乔纳斯·泰勒（杰森·斯坦森饰）与科学家张九溟（吴京饰）双雄联手，进入海底7000米深渊执行探索任务。他们意外遭遇史前巨兽海洋霸主巨齿鲨群的攻击，还将对战凶猛危险的远古怪兽群。惊心动魄的深渊冒险，巨燃巨爽的深海大战一触即发。",
        description="答案"
    )
    evidences: List[str] = Field(
        [
            "海洋霸主巨齿鲨，今夏再掀狂澜！乔纳斯·泰勒（杰森·斯坦森 饰）与科学家张九溟（吴京 饰）双雄联手，进入海底7000米深渊执行探索任务。他们意外遭遇史前巨兽海洋霸主巨齿鲨群的攻击，还将对战凶猛危险的远古怪兽群。惊心动魄的深渊冒险，巨燃巨爽的深海大战一触即发",
            "本·维特利 编剧：乔·霍贝尔埃里希·霍贝尔迪恩·乔格瑞斯 国家地区：中国 | 美国 发行公司：上海华人影业有限公司五洲电影发行有限公司中国电影股份有限公司北京电影发行分公司 出品公司：上海华人影业有限公司华纳兄弟影片公司北京登峰国际文化传播有限公司 更多片名：巨齿鲨2 剧情：海洋霸主巨齿鲨，今夏再掀狂澜！乔纳斯·泰勒（杰森·斯坦森 饰）与科学家张九溟（吴京 饰）双雄联手，进入海底7000米深渊执行探索任务。他们意外遭遇史前巨兽海洋霸主巨齿鲨群的攻击，还将对战凶猛危险的远古怪兽群。惊心动魄的深渊冒险，巨燃巨爽的深海大战一触即发……"
        ],
        description="待引用的检索文档")
    selected_idx: List[int] = Field(
        [1,2],
        description="文档对应的索引"
    )
    show_code:bool=Field(
        default=True,
        description="是否显示引用代码块"
    )
    show_summary: bool = Field(
        default=False,
        description="是否使用原文"
    )
    selected_docs:List[dict]=Field(
        default=[]

    )