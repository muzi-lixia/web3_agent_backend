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
