"""Evidence contract: exact source quotes, cautious translations and numeric safety."""
import json
import pytest
from web3_agent.rag.answer import grounded_answer, _normalized
from web3_agent.agent.graph.read_graph import initial_state
from web3_agent.config.constants import Intent


def hit(content, key='S1'):
    return dict(id=key, title='规则', section='FAQ', version='v1', kb_id='default',
                doc_category='faq', updated_at='2026-01-01', content=content, source_url='')


def draft(text, quote, key='S1'):
    return json.dumps({'claims':[{'text':text,'source_id':key,'quote':quote}]},ensure_ascii=False)


@pytest.mark.parametrize(('text','expected'), [('七天','7天'),('十二天','12天'),
    ('三十天','30天'),('二零二六年','2026年'),('一百天','一百天')])
def test_conservative_numerals(text, expected):
    assert _normalized(text) == expected


@pytest.mark.parametrize('wrong', ['锁仓二零二六年。[S1]','锁仓七点六年。[S1]','锁仓十六年。[S1]'])
def test_wrong_numbers_never_equal_six(wrong):
    assert grounded_answer(wrong,[hit('锁仓6年。')])[1]


def test_chat_discards_all_model_generated_scope_fields():
    state = initial_state([], '什么是DEX？',intent=Intent.QUERY_KNOWLEDGE,
        slots={'protocol_id':'通用','doc_category':'DEX','chain_id':56},kb_id='guides')
    assert state['rag_args'] == {'query_str':'什么是DEX？','kb_id':'guides'}


def test_checked_unlock_translation_preserves_quote_and_chinese():
    quote = 'The unlock period is 7 days.'
    result, degraded = grounded_answer(draft('解锁期为7天。',quote),[hit(quote)])
    assert not degraded and '解锁期为7天。 [S1]' in result and quote in result
    assert 'claims' not in result


def test_wrong_translated_number_is_replaced_by_related_quote_only():
    quote = 'The unlock period is 7 days.'
    result, degraded = grounded_answer(draft('解锁期为6天。',quote),[hit(quote)])
    assert degraded and '解锁期为6天' not in result and quote in result


def test_sensitive_translation_with_conditions_is_not_auto_certified():
    quote = 'Early exit within the lockup window incurs a penalty of 2%.'
    result, degraded = grounded_answer(draft('提前退出罚金为2%。',quote),[hit(quote+' Unrelated FAQ text.')])
    assert degraded and quote in result and 'Unrelated FAQ' not in result
    assert '提前退出罚金为2%' not in result


@pytest.mark.parametrize(('quote','key'), [('penalty of 2%.','S1'),
    ('Invented rule.','S1'),('The unlock period is 7 days.','S2')])
def test_incomplete_fabricated_or_wrong_source_quote_is_rejected(quote,key):
    source = 'Early exit within the lockup window incurs a penalty of 2%.'
    result, degraded = grounded_answer(draft('解锁期为7天。',quote,key),[hit(source)])
    assert degraded and '未通过证据校验' in result and quote not in result


def test_non_sensitive_translation_renders_only_bound_source():
    quote = 'DEX means decentralized exchange.'
    result, degraded = grounded_answer(draft('DEX 是去中心化交易所。',quote),
                                       [hit(quote),hit('Unrelated content.','S2')])
    assert not degraded and 'DEX 是去中心化交易所' in result and quote in result
    assert 'Unrelated content' not in result and '[S2]' not in result


def test_malformed_structured_response_cannot_leak_json():
    result, degraded = grounded_answer('{"claims": "bad"}',[hit('source')])
    assert degraded and 'claims' not in result
