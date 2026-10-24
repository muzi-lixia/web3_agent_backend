"""Partial failures must not masquerade as empty wallets or erase valid results."""
import asyncio
import importlib
from types import SimpleNamespace
from unittest.mock import AsyncMock
import httpx
import pytest
from web3_agent.config.settings import Settings
from web3_agent.agent.render import render_transactions
from web3_agent.agent.tool_registry import ToolRegistry
from web3_agent.agent.tool_runner import ToolRunner
from web3_agent.config.constants import CallSource, ToolGroup

transactions = importlib.import_module('web3_agent.web3.tools.read.get_transaction_records')
assets = importlib.import_module('web3_agent.web3.tools.read.get_asset_balance')
ADDRESS = '0x' + '1'*40


def test_transaction_all_sources_failed_is_not_empty_and_runner_marks_failure(monkeypatch):
    async def run():
        monkeypatch.setattr(transactions,'_fetch_source',AsyncMock(side_effect=TimeoutError))
        tool=transactions.build_get_transaction_records(
            Settings(_env_file=None,explorer_api_key='test',allowed_chain_ids=[1]))
        registry=ToolRegistry()
        registry.register(tool,ToolGroup.LLM_VISIBLE,mutating=False)
        import json
        content, record=await ToolRunner(registry).run(tool.name,{'address':ADDRESS},source=CallSource.LLM)
        result=json.loads(content)
        assert result['status']=='failed' and result['error']=='E302' and not record.ok
        assert len(result['errors'])==3
        assert '没有查到' not in render_transactions(result)
    asyncio.run(run())


@pytest.mark.parametrize('failed',[False,True])
def test_transaction_empty_is_qualified_if_any_source_failed(monkeypatch,failed):
    async def run():
        async def fetch(*args):
            if failed and args[-1]=='tokentx':
                raise TimeoutError
            return []
        monkeypatch.setattr(transactions,'_fetch_source',fetch)
        tool=transactions.build_get_transaction_records(
            Settings(_env_file=None,explorer_api_key='test',allowed_chain_ids=[1]))
        result=await tool.ainvoke({'address':ADDRESS})
        assert result['status']==('partial' if failed else 'empty')
        text=render_transactions(result)
        if failed:
            assert '成功查询的范围' in text and '结果不完整' in text and 'tokentx' in text
        else:
            assert '没有查到交易记录' in text and not result['errors']
    asyncio.run(run())


def test_partial_transaction_keeps_successful_records_and_warns(monkeypatch):
    async def run():
        async def fetch(*args):
            if args[-1]!='txlist':
                raise TimeoutError('internal URL and API key must not leak')
            return [{'hash':'0xabc','timeStamp':'1','value':'0','from':ADDRESS,'to':ADDRESS}]
        monkeypatch.setattr(transactions,'_fetch_source',fetch)
        tool=transactions.build_get_transaction_records(
            Settings(_env_file=None,explorer_api_key='test',allowed_chain_ids=[1]))
        result=await tool.ainvoke({'address':ADDRESS})
        assert result['status']=='partial' and len(result['records'])==1
        assert '0xabc' in render_transactions(result) and '结果不完整' in render_transactions(result)
        assert 'API key' not in str(result)
    asyncio.run(run())


@pytest.mark.parametrize(('body','empty'),[
    ({'status':'0','message':'No transactions found','result':[]},True),
    ({'status':'0','message':'No transactions found.','result':[]},True),
    ({'status':'0','message':'No records found','result':[]},True),
    ({'status':'1','message':'OK','result':[]},True),
    ({'status':'0','message':'NOTOK','result':'Max rate limit reached'},False),
    ({'status':'0','message':'NOTOK','result':[]},False),
])
def test_explorer_http_200_errors_are_not_empty_results(body,empty):
    async def run():
        client=SimpleNamespace(get=AsyncMock(return_value=httpx.Response(200,json=body,
            request=httpx.Request('GET','http://test'))))
        if empty:
            assert await transactions._fetch_source(client,'http://test','test',1,ADDRESS,'txlist') == []
        else:
            with pytest.raises(ValueError):
                await transactions._fetch_source(client,'http://test','test',1,ADDRESS,'txlist')
    asyncio.run(run())


@pytest.mark.parametrize('partial',[False,True])
def test_asset_all_failed_vs_partial_and_zero_is_valid_data(monkeypatch,partial):
    async def run():
        async def query(adapter,chain_id,address,top_n):
            if partial and chain_id==1:
                return {'balances':[{'chainId':1,'raw':'0','human':'0'}],'errors':[]}
            raise TimeoutError('private endpoint')
        monkeypatch.setattr(assets,'_query_chain',query)
        tool=assets.build_get_asset_balance(SimpleNamespace(available_chains=lambda:[1,56]))
        result=await tool.ainvoke({'address':ADDRESS})
        if partial:
            assert result['status']=='partial' and result['balances'][0]['raw']=='0' and 'error' not in result
        else:
            assert result['status']=='failed' and result['error']=='E301' and result['balances']==[]
        assert 'private endpoint' not in str(result)
    asyncio.run(run())


def test_cancelled_source_propagates_cancellation(monkeypatch):
    async def run():
        monkeypatch.setattr(transactions,'_fetch_source',AsyncMock(side_effect=asyncio.CancelledError))
        tool=transactions.build_get_transaction_records(
            Settings(_env_file=None,explorer_api_key='test',allowed_chain_ids=[1]))
        with pytest.raises(asyncio.CancelledError):
            await tool.ainvoke({'address':ADDRESS})
    asyncio.run(run())


@pytest.mark.parametrize('all_dirty',[False,True])
def test_bad_transaction_records_do_not_discard_good_records(monkeypatch,all_dirty):
    async def run():
        async def fetch(*args):
            bad=[{}, {'hash':'bad-time','timeStamp':'broken'}, None]
            good=[{'hash':'0x'+args[-1], 'timeStamp':'1','value':'0'}]
            return bad if all_dirty else good+bad
        monkeypatch.setattr(transactions,'_fetch_source',fetch)
        tool=transactions.build_get_transaction_records(
            Settings(_env_file=None,explorer_api_key='test',allowed_chain_ids=[1]))
        result=await tool.ainvoke({'address':ADDRESS})
        if all_dirty:
            assert result['status']=='failed' and result['error']=='E302'
        else:
            assert result['status']=='partial' and len(result['records'])==3
            assert 'error' not in result
        assert sum(e['dropped_records'] for e in result['errors'])==9
    asyncio.run(run())


def test_asset_failure_keeps_internal_trace_in_logs_only(monkeypatch,caplog):
    from contextlib import asynccontextmanager
    from web3_agent.common.exceptions import RpcUnavailable
    async def run():
        @asynccontextmanager
        async def session():
            yield object()
        monkeypatch.setattr(assets,'session_scope',session)
        monkeypatch.setattr(assets.token_registry_repo,'list_whitelisted',AsyncMock(return_value=[]))
        adapter=SimpleNamespace(get_native_balance=AsyncMock(side_effect=RpcUnavailable('internal endpoint 401')))
        result=await assets._query_chain(adapter,56,ADDRESS,20)
        assert 'internal endpoint' not in str(result)
        assert 'chain_id=56' in caplog.text and 'internal endpoint 401' in caplog.text
        assert any(record.exc_info for record in caplog.records)
    asyncio.run(run())


def test_direct_renderer_preserves_specific_error_message():
    assert render_transactions({'error':'E101','message':'地址格式不合法'}) == '地址格式不合法'
