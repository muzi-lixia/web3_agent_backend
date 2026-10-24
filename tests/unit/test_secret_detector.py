from web3_agent.agent.secret_detector import detect


def test_private_key_detected() -> None:
    text = "我的私钥是 0x" + "a1" * 32
    hit = detect(text)
    assert hit is not None
    assert hit.kind == "private_key"
    assert "a1a1a1" not in hit.excerpt


def test_bare_64_hex_detected_without_prefix() -> None:
    hit = detect("key: " + "b" * 64)
    assert hit is not None
    assert hit.kind == "private_key"


def test_twelve_words_detected_as_mnemonic() -> None:
    words = "abandon ability able about above absent absorb abstract absurd abuse access accident"
    hit = detect(words)
    assert hit is not None
    assert hit.kind == "mnemonic"


def test_normal_sentence_passes() -> None:
    assert detect("帮我查一下 ETH 余额") is None


def test_transaction_hash_is_not_a_private_key() -> None:
    """交易哈希就是 0x + 64 位 hex，和私钥同形。没有「私钥」这类提示时不能拦。"""
    assert detect("看看这笔交易 0x" + "c3" * 32) is None


def test_plain_english_sentence_is_not_a_mnemonic() -> None:
    text = ("Please send the payment to the address below and then we can "
            "continue the discussion tomorrow")
    assert detect(text) is None


def test_sentence_with_punctuation_is_not_a_mnemonic() -> None:
    assert detect("the quick brown fox jumps over the lazy dog, and so on") is None


def test_mnemonic_on_its_own_line_is_caught() -> None:
    words = "abandon ability able about above absent absorb abstract absurd abuse access accident"
    hit = detect(f"帮我看看这个\n{words}\n能用吗")
    assert hit is not None and hit.kind == "mnemonic"
