"""RAG responses are summarized by the model, not rewritten by literal checks."""
import pytest
from web3_agent.rag.answer import render_answer
from web3_agent.agent.graph.read_graph import initial_state
from web3_agent.agent.prompts import RAG_SYSTEM
from web3_agent.config.constants import Intent


def hit(content):
    return dict(id="S1", title="overview", section="正文", version="v1", kb_id="default",
                doc_category="faq", updated_at="2026-01-01", content=content, source_url="")


@pytest.mark.parametrize("draft", [
    "Berachain 是兼容 EVM 的第 1 层区块链，区块时间约 2 秒。[S1]",
    "解锁期为七天。[S1]",
    "提前退出可能产生罚金，具体条件以文档为准。[S1]",
    "Berachain is an EVM-compatible layer 1 blockchain.[S1]",
])
def test_natural_language_summary_is_preserved(draft):
    answer = render_answer(draft, [hit("English source material.")])
    assert answer.startswith(draft)
    assert "未通过证据校验" not in answer and "无法自动核实" not in answer
    assert answer == draft
    assert "来源：" not in answer and "overview" not in answer


def test_source_content_is_not_dumped_into_answer():
    answer = render_answer("DEX 是去中心化交易所。",[hit("Unrelated source paragraph.")])
    assert "Unrelated source paragraph" not in answer


def test_chat_discards_all_model_generated_scope_fields():
    state = initial_state([], "什么是DEX？", intent=Intent.QUERY_KNOWLEDGE,
        slots={"protocol_id":"通用","doc_category":"DEX","chain_id":56},kb_id="guides")
    assert state["rag_args"] == {"query_str":"什么是DEX？","kb_id":"guides"}


def test_prompt_requests_user_language_and_plain_answer():
    assert "中文提问用中文，英文提问用英文" in RAG_SYSTEM
    assert "准确翻译" in RAG_SYSTEM
    assert "前端独立展示" in RAG_SYSTEM
    assert "只输出 JSON" not in RAG_SYSTEM


def test_history_keeps_citations_separate_in_existing_json_metadata():
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from web3_agent.agent.agent_loop import AgentLoop
    from web3_agent.common.types import ChatOutcome, ToolCallRecord
    source = hit('source paragraph')
    outcome = ChatOutcome(kind='answer',content='中文回答。[S1]',citations=[source],
                          tool_calls=[ToolCallRecord(name='rag_general_knowledge',ok=True)])
    conversation = SimpleNamespace(save=AsyncMock())
    instance = SimpleNamespace(_conversation=conversation)
    result = asyncio.run(AgentLoop._finish(instance,None,'user',outcome))
    saved = conversation.save.call_args.kwargs
    assert saved['content'] == '中文回答。[S1]'
    assert saved['tool_calls'][0]['citations'] == [source]
    assert result.citations == [source]
