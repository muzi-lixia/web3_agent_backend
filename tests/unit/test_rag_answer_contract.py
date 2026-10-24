"""RAG responses are summarized by the model, not rewritten by literal checks."""
from web3_agent.agent.graph.read_graph import initial_state
from web3_agent.agent.prompts import RAG_SYSTEM
from web3_agent.config.constants import Intent


def hit(content):
    return dict(id="S1", title="overview", section="正文", version="v1", kb_id="default",
                doc_category="faq", updated_at="2026-01-01", content=content, source_url="")


def test_chat_discards_all_model_generated_scope_fields():
    """检索参数只由程序给：问题原文 + 知识库 id，模型填的过滤条件一个都不进来。"""
    state = initial_state([], "什么是DEX？", intent=Intent.QUERY_KNOWLEDGE, kb_id="guides")
    assert state["rag_args"] == {"query_str":"什么是DEX？","kb_id":"guides"}


def test_prompt_requests_user_language_and_plain_answer():
    assert "中文提问用中文，英文提问用英文" in RAG_SYSTEM
    assert "准确翻译" in RAG_SYSTEM
    assert "前端独立展示" in RAG_SYSTEM
    assert "只输出 JSON" not in RAG_SYSTEM


def test_history_keeps_citations_separate_in_existing_json_metadata(monkeypatch):
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock
    from web3_agent.agent import agent_loop as agent_loop_module
    from web3_agent.agent.agent_loop import AgentLoop
    from web3_agent.common.types import ChatOutcome, ToolCallRecord

    class _Scope:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *exc_info):
            return False

    async def _fresh(*args, **kwargs):
        return True

    save = AsyncMock(return_value=0)
    monkeypatch.setattr(agent_loop_module, "session_scope", lambda: _Scope())
    monkeypatch.setattr(agent_loop_module.conversation_repo, "assert_generation", _fresh)
    monkeypatch.setattr(agent_loop_module.conversation_repo, "append_message", save)
    source = hit('source paragraph')
    outcome = ChatOutcome(kind='answer',content='中文回答。[S1]',citations=[source],
                          tool_calls=[ToolCallRecord(name='rag_general_knowledge',ok=True)])
    instance = SimpleNamespace()
    result = asyncio.run(
        AgentLoop._finish(instance, 'user', outcome, generation=0, turn_id='t-1')
    )
    saved = save.call_args.kwargs
    assert saved['content'] == '中文回答。[S1]'
    assert saved['tool_calls'][0]['citations'] == [source]
    assert saved['turn_id'] == 't-1'
    assert result.citations == [source]
