"""Build the answer-only context independently of raw tool output and chat history."""
from langchain_core.messages import HumanMessage, SystemMessage
from web3_agent.agent.prompts import RAG_SYSTEM


def answer_messages(question: str, citations: list[dict], context: str = ""):
    materials = "\n\n".join(
        f"【{hit['id']}】\n{hit['content']}" for hit in citations
    )
    history = (
        f"【对话上下文】\n此前用户问题：\n{context}\n\n"
        if context else ""
    )
    prompt = (
        f"{history}【用户问题】\n{question}\n\n"
        f"【检索材料】\n"
        f"<retrieved_materials>\n{materials}\n</retrieved_materials>"
    )
    return [SystemMessage(content=RAG_SYSTEM), HumanMessage(content=prompt)]
