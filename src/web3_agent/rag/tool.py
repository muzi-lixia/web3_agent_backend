"""与链上查询工具平级的通用只读文档检索工具。"""
from langchain_core.tools import StructuredTool


def build_rag_general_knowledge(retriever):
    async def rag_general_knowledge(
        query_str: str, chain_id: int | None = None,
        protocol_id: str | None = None,
        doc_category: str | None = None,
        top_k: int = 3,
        kb_id: str = "default",
        conversation_context: str = ""
    ) -> dict:
        """检索 Web3 静态知识库：协议规则、概念、教程、钱包、Gas、NFT、DEX、FAQ、风险和报错解释。
        kb_id 限定当前知识库（默认 default），绝不跨库补全；chain_id、protocol_id、doc_category 为可选精确过滤。
        不查询实时余额、APY、奖励或交易记录，不构造交易，不校验交易参数。
        只有用户或上下文明示的范围才填写；不要猜测协议或链。"""
        return await retriever.search(
            query_str,
            chain_id=chain_id,
            protocol_id=protocol_id,
            doc_category=doc_category,
            top_k=top_k,
            kb_id=kb_id,
            conversation_context=conversation_context
        )
    return StructuredTool.from_function(
        coroutine=rag_general_knowledge,
        name="rag_general_knowledge",
        description=rag_general_knowledge.__doc__
    )
