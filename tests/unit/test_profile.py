"""偏好提取与渲染：只认显式偏好，渲染不超预算。纯函数，不碰库、不调模型。"""

from __future__ import annotations

from web3_agent.domain.memory import profile


def test_plain_complaint_is_not_a_preference() -> None:
    assert profile.extract("这个回答太长了").is_empty


def test_explicit_style_is_recorded() -> None:
    patch = profile.extract("以后都用简洁的回答")
    assert patch.answer_style == "concise"


def test_detailed_style() -> None:
    assert profile.extract("以后详细一点").answer_style == "detailed"


def test_language() -> None:
    assert profile.extract("以后都用英文回答").language == "en"
    assert profile.extract("以后都用中文").language == "zh"


def test_chain_alias_needs_chain_word() -> None:
    assert profile.extract("以后都用 berachain 链").preferred_chain_id == 80094
    # 没有「链」字样时，同名的代币不该被当成链
    assert profile.extract("以后都查 BERA").preferred_chain_id is None


def test_risk_level() -> None:
    assert profile.extract("以后谨慎一点").risk_preference == "conservative"


def test_token_by_ticker() -> None:
    assert profile.extract("以后都查 BERA").tokens == ("BERA",)


def test_lowercase_unknown_word_is_not_a_token() -> None:
    assert profile.extract("以后都查 dao").tokens == ()


def test_multiple_patches_in_one_sentence() -> None:
    patch = profile.extract("以后都用英文、简洁回答，默认查 USDT")
    assert patch.answer_style == "concise"
    assert patch.language == "en"
    assert patch.tokens == ("USDT",)


def test_render_empty_profile_returns_empty_string() -> None:
    assert profile.render(profile.Profile(), max_chars=400, max_tokens=5) == ""


def test_render_contains_labels() -> None:
    text = profile.render(
        profile.Profile(
            answer_style="concise",
            language="en",
            preferred_chain_id=80094,
            frequent_tokens=({"chainId": 80094, "symbol": "BERA"},),
        ),
        max_chars=400,
        max_tokens=5,
    )
    assert profile.PROFILE_HEADER in text
    assert "简洁" in text and "英文" in text
    assert "80094" in text and "BERA" in text


def test_render_drops_tail_when_over_budget() -> None:
    text = profile.render(
        profile.Profile(
            answer_style="concise",
            language="en",
            preferred_chain_id=80094,
            frequent_tokens=tuple(
                {"chainId": 80094, "symbol": f"T{i}"} for i in range(5)
            ),
        ),
        max_chars=80,
        max_tokens=5,
    )
    lines = text.splitlines()
    assert lines[0] == profile.PROFILE_HEADER
    assert len(lines) < 5
    assert len(text) <= 80
