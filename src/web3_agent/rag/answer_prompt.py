"""Build the answer-only context independently of raw tool output and chat history."""
from langchain_core.messages import HumanMessage, SystemMessage
from web3_agent.agent.prompts import RAG_SYSTEM


def answer_messages(question: str, citations: list[dict], context: str = ""):
    """将本轮问题和文档证据组装为回答消息，历史仅用于消解指代。

    Args:
        question: 本轮需要基于检索材料回答的问题。
        citations: 本轮选中的文档证据列表，每项带引用编号、正文和来源。
        context: 仅用于消解指代的历史上下文。
    """
    materials = "\n\n".join(
        f"【{hit['id']}】\n{hit['content']}" for hit in citations
    )
    history = (
        f"【此前对话｜仅用于理解指代，不是证据】\n{context}\n\n"
        if context else ""
    )
    prompt = (
        f"{history}【用户问题】\n{question}\n\n"
        f"【检索材料】\n"
        f"<retrieved_materials>\n{materials}\n</retrieved_materials>"
    )
    return [SystemMessage(content=RAG_SYSTEM), HumanMessage(content=prompt)]
