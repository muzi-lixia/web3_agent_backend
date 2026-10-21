"""EIP-55 校验和。向量全部取自 EIP-55 原文。"""

from web3_agent.common.evm_address import normalize_address, to_checksum_address

EIP55_VALID = (
    "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAed",
    "0xfB6916095ca1df60bB79Ce92cE3Ea74c37c5d359",
    "0xdbF03B407c01E7cD3CBea99509d93f8DDDC8C6FB",
    "0xD1220A0cf47c7B9Be7A2E6BA89F429762e7b9aDb",
)

EIP55_INVALID_CASE = (
    "0x5aAeb6053F3E94C9b9A09f33669435E7Ef1BeAeD",  # 末位大小写被改动
    "0x5AAeb6053F3E94C9b9A09f33669435E7Ef1BeAed",  # 第三位大小写被改动
)


def test_eip55_checksum_matches_spec() -> None:
    for address in EIP55_VALID:
        assert to_checksum_address(address) == address


def test_normalize_accepts_valid_checksum() -> None:
    for address in EIP55_VALID:
        assert normalize_address(address) == address.lower()


def test_normalize_accepts_all_lowercase() -> None:
    address = EIP55_VALID[0]
    assert normalize_address(address.lower()) == address.lower()


def test_normalize_rejects_wrong_checksum() -> None:
    for address in EIP55_INVALID_CASE:
        assert normalize_address(address) is None


def test_normalize_rejects_malformed() -> None:
    assert normalize_address("0x" + "1" * 39) is None
    assert normalize_address("0x" + "z" * 40) is None
    assert normalize_address(EIP55_VALID[0][:-2]) is None
    assert normalize_address("0x" + "0" * 40) is None  # 零地址永远不是有效目标
