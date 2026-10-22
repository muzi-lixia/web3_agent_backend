"""知识库作用域参数；链、协议、分类均可省略，省略即不限制该维度。"""
import re


def normalize_filters(chain_id=None, protocol_id=None, doc_category=None, kb_id="default"):
    if not isinstance(kb_id,str) or not re.fullmatch(r"[A-Za-z0-9_.:-]{1,64}",kb_id):
        raise ValueError("知识库标识格式错误")
    if chain_id is not None:
        if isinstance(chain_id, bool) or not re.fullmatch(r"[1-9][0-9]*", str(chain_id)):
            raise ValueError("chain_id 必须为正整数")
        chain_id = int(chain_id)
        if chain_id > 2147483647:
            raise ValueError("chain_id 超出范围")
    for value, pattern in ((protocol_id, r"[A-Za-z0-9_.:-]{1,128}"),
                           (doc_category, r"[a-z][a-z0-9_]{0,63}")):
        if value is not None and (not isinstance(value, str) or not re.fullmatch(pattern, value)):
            raise ValueError("协议或分类标签格式错误")
    return dict(chain_id=chain_id, protocol_id=protocol_id, doc_category=doc_category, kb_id=kb_id)
