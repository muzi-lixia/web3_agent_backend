"""记忆：中期摘要（V1-b）与结构化档案（V1-c）。自由文本向量记忆排 V2。

这里不 re-export Summarizer：它依赖 agent.llm_adapter，而 llm_adapter 又要用 schema，
在包初始化时交叉导入会成环。使用方直接 import 子模块。
"""
