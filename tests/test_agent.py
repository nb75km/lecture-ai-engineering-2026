"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

from pydantic_ai import DeferredToolRequests
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app.agent import agent, resolve_approvals, run_conversation

BASE_TOOLS = {"search_glossary", "get_lecture_summary"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    model = TestModel(call_tools=sorted(BASE_TOOLS))
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    assert len(_parts(result, ToolReturnPart)) == len(_parts(result, ToolCallPart))
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


def test_learning_log_waits_for_approval_and_only_saves_when_approved(tmp_path, monkeypatch):
    from app import tools

    monkeypatch.setattr(tools, "LEARNING_LOG_DIR", tmp_path / "learning_log")
    model = TestModel(call_tools=["save_learning_log"])
    with agent.override(model=model):
        pending = agent.run_sync("今日の学びを学習記録に保存してください")

    assert isinstance(pending.output, DeferredToolRequests)
    assert len(pending.output.approvals) == 1
    call = pending.output.approvals[0]
    assert call.tool_name == "save_learning_log"
    assert not tools.LEARNING_LOG_DIR.exists()

    monkeypatch.setattr("builtins.input", lambda _: "n")
    denied = resolve_approvals(agent, pending, TestModel(call_tools=[]))
    assert isinstance(denied.output, str)
    assert not tools.LEARNING_LOG_DIR.exists()

    with agent.override(model=TestModel(call_tools=[])):
        approved = agent.run_sync(
            message_history=pending.all_messages(),
            deferred_tool_results=pending.output.build_results(
                approvals={call.tool_call_id: True}
            ),
        )
    assert isinstance(approved.output, str)
    saved_files = list(tools.LEARNING_LOG_DIR.glob("*.md"))
    assert len(saved_files) == 1
    assert saved_files[0].read_text(encoding="utf-8") == "a"


def test_conversation_waits_for_and_sends_user_reply(monkeypatch):
    from unittest.mock import Mock

    first_result = Mock()
    first_result.output = "保存してもよろしいでしょうか？"
    first_result.all_messages.return_value = ["assistant question"]
    final_result = Mock()
    final_result.output = "承知しました。"
    target = Mock()
    target.run_sync.side_effect = [first_result, final_result]
    monkeypatch.setattr("builtins.input", lambda _: "はい")

    result = run_conversation(target, "学びを記録してください", model=object())

    assert result is final_result
    assert target.run_sync.call_count == 2
    assert target.run_sync.call_args_list[1].args[0] == "はい"
    assert target.run_sync.call_args_list[1].kwargs["message_history"] == ["assistant question"]
