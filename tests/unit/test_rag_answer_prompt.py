"""Only minimal, numbered document material reaches the RAG answer model."""
import asyncio
import json
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from web3_agent.agent.graph.read_graph import build_read_graph, initial_state
from web3_agent.common.types import ToolCallRecord
from web3_agent.config.constants import Intent
from web3_agent.rag.answer_prompt import answer_messages


def test_material_prompt_contains_question_content_and_stable_ids_only():
    messages = answer_messages('什么是Berachain？', [dict(id='S1',content='Berachain is an EVM chain.',
        title='SECRET_TITLE',source_url='https://internal/source',distance=.1,version='private-version')])
    assert isinstance(messages[0],SystemMessage) and isinstance(messages[1],HumanMessage)
    content = messages[1].content
    assert '【用户问题】\n什么是Berachain？' in content
    assert '【S1】\nBerachain is an EVM chain.' in content
    assert all(value not in content for value in ['SECRET_TITLE','https://internal','private-version','distance'])
    assert '根据用户当前输入的语言回答' in messages[0].content
    assert '不输出来源列表' in messages[0].content
    assert '使用用户当前提问的语言' not in content
    assert '请依据以上相关材料' not in content
    assert '不输出来源列表' not in content


def test_graph_keeps_raw_tool_trace_but_excludes_it_and_old_sources_from_model_input():
    seen=[]
    hit=dict(id='S1',chunk_id='chunk1',content='Berachain is an EVM chain.',title='overview',
             version='v1',updated_at='SECRET_TIMESTAMP',distance=.2)
    class Model:
        async def ainvoke(self,messages):
            seen.extend(messages)
            return AIMessage(content='Berachain 是兼容 EVM 的区块链。[S1]')
    class Runner:
        async def run(self,name,args,source):
            payload=dict(status='success',hits=[hit],retrieval={'fusion_scores':{'chunk1':.1}})
            return json.dumps(payload),ToolCallRecord(name=name,args=args,ok=True)
    state=initial_state([('user','什么是Berachain？'),('assistant','旧回答\n来源：STALE_SOURCE')],
                        '它兼容EVM吗？',intent=Intent.QUERY_KNOWLEDGE)
    result=asyncio.run(build_read_graph(None,Runner(),5,rag_answer_model=Model()).ainvoke(state))
    assert len(seen)==2
    model_input='\n'.join(str(m.content) for m in seen)
    assert '什么是Berachain？' in model_input and '它兼容EVM吗？' in model_input
    assert '【S1】\nBerachain is an EVM chain.' in model_input
    assert all(s not in model_input for s in ['STALE_SOURCE','SECRET_TIMESTAMP','fusion_scores'])
    assert not any(isinstance(m,ToolMessage) for m in seen)
    tool_messages=[m for m in result['messages'] if isinstance(m,ToolMessage)]
    assert 'fusion_scores' in tool_messages[0].content
    assert result['rag_citations'][0]['updated_at']=='SECRET_TIMESTAMP'


def test_context_is_optional_and_remains_in_data_message():
    messages = answer_messages('它是什么？', [{'id':'S1','content':'English material.'}], '此前讨论 DEX')
    assert '【此前对话｜仅用于理解指代，不是证据】' in messages[1].content
    assert '此前讨论 DEX' not in messages[0].content
    assert '【对话上下文】' not in answer_messages('DEX是什么？',[])[1].content
