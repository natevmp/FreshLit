from __future__ import annotations

import copy
import io
import logging
import os
import subprocess
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator
from unittest.mock import MagicMock, patch

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
from openai_codex import ApprovalMode, Sandbox, ServerBusyError, TurnHandle
from openai_codex.generated.v2_all import (
    AgentMessageThreadItem,
    ThreadItem,
    Turn,
    TurnStatus,
)
from openai_codex.models import (
    ItemCompletedNotification,
    Notification,
    TurnCompletedNotification,
)
from pydantic import BaseModel, Field, ValidationError

from freshlit.utils import llm
from freshlit.utils.config import LLMConfig, Settings


class NestedOutput(BaseModel):
    label: str = "default-label"


class OutputModel(BaseModel):
    value: int
    nested: NestedOutput = Field(default_factory=NestedOutput)


class MapOutput(BaseModel):
    values: dict[str, int]


def completed_result(response: str, item_type: str = "agentMessage") -> SimpleNamespace:
    item = SimpleNamespace(root=SimpleNamespace(type=item_type))
    return SimpleNamespace(
        status="completed", final_response=response, items=[item]
    )


def typed_turn_events(
    response: str,
    *,
    thread_id: str = "thread-typed",
    turn_id: str = "turn-typed",
) -> tuple[ThreadItem, Notification, Notification]:
    item = ThreadItem(
        root=AgentMessageThreadItem(
            id="item-typed", text=response, type="agentMessage"
        )
    )
    item_notification = Notification(
        method="item/completed",
        payload=ItemCompletedNotification(
            completedAtMs=1,
            item=item,
            threadId=thread_id,
            turnId=turn_id,
        ),
    )
    completion = Notification(
        method="turn/completed",
        payload=TurnCompletedNotification(
            threadId=thread_id,
            turn=Turn(
                id=turn_id,
                items=[item],
                status=TurnStatus.completed,
            ),
        ),
    )
    return item, item_notification, completion


class FakeProcess:
    def __init__(self) -> None:
        self.status: int | None = None
        self.terminateCalls = 0
        self.killCalls = 0

    def poll(self) -> int | None:
        return self.status

    def wait(self, timeout: float | None = None) -> int:
        if self.status is None:
            raise subprocess.TimeoutExpired("fake-codex", timeout)
        return self.status

    def terminate(self) -> None:
        self.terminateCalls += 1

    def kill(self) -> None:
        self.killCalls += 1
        self.status = -9


class StrictSchemaTests(unittest.TestCase):
    def test_recursive_normalization_is_strict_and_does_not_mutate(self) -> None:
        original = OutputModel.model_json_schema()
        before = copy.deepcopy(original)
        normalized = llm._openai_strict_json_schema(original)

        self.assertEqual(original, before)
        self.assertEqual(normalized["required"], ["value", "nested"])
        self.assertFalse(normalized["additionalProperties"])
        nested = normalized["$defs"]["NestedOutput"]
        self.assertEqual(nested["required"], ["label"])
        self.assertFalse(nested["additionalProperties"])

        def assert_no_defaults(node: object) -> None:
            if isinstance(node, dict):
                self.assertNotIn("default", node)
                for child in node.values():
                    assert_no_defaults(child)
            elif isinstance(node, list):
                for child in node:
                    assert_no_defaults(child)

        assert_no_defaults(normalized)

    def test_map_schemas_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "map-like"):
            llm._openai_strict_json_schema(MapOutput.model_json_schema())
        with self.assertRaisesRegex(ValueError, "map-like"):
            llm._openai_strict_json_schema(
                {"type": "object", "additionalProperties": True}
            )


class CodexLowLevelSafetyTests(unittest.TestCase):
    def test_early_turn_completion_is_replayed_after_registration(self) -> None:
        client = llm._build_codex_client(llm.CodexConfig())
        item, item_notification, completion = typed_turn_events(
            '{"value":7,"nested":{"label":"typed"}}',
            thread_id="thread-early",
            turn_id="turn-early",
        )

        client._router.route_notification(item_notification)
        client._router.route_notification(completion)
        client._router.register_turn("turn-early")
        self.assertEqual(
            client._router._turn_notifications["turn-early"].qsize(), 2
        )
        result = TurnHandle(client, "thread-early", "turn-early").run()

        self.assertEqual(result.items, [item])
        self.assertEqual(
            result.final_response,
            '{"value":7,"nested":{"label":"typed"}}',
        )

    def test_turn_registration_cannot_reorder_replay_and_live_completion(self) -> None:
        replayEntered = threading.Event()
        replayRelease = threading.Event()
        liveStarted = threading.Event()
        liveFinished = threading.Event()
        router = llm._EarlyCompletionMessageRouter()
        _, item_notification, completion = typed_turn_events(
            "ordered", turn_id="turn-interleaved"
        )

        class BlockingPending:
            def __iter__(self) -> Iterator[Notification]:
                replayEntered.set()
                if not replayRelease.wait(0.2):
                    raise RuntimeError("bounded replay release timed out")
                yield item_notification

        router._pending_turn_notifications["turn-interleaved"] = BlockingPending()
        workerOutcome: dict[str, BaseException] = {}

        def register() -> None:
            try:
                router.register_turn("turn-interleaved")
            except BaseException as exc:
                workerOutcome["register"] = exc

        def route_live_completion() -> None:
            liveStarted.set()
            try:
                router.route_notification(completion)
            except BaseException as exc:
                workerOutcome["route"] = exc
            finally:
                liveFinished.set()

        registerThread = threading.Thread(target=register)
        liveThread = threading.Thread(target=route_live_completion)
        registerThread.start()
        self.assertTrue(replayEntered.wait(0.2))
        liveThread.start()
        self.assertTrue(liveStarted.wait(0.2))
        self.assertFalse(liveFinished.wait(0.02))
        replayRelease.set()
        registerThread.join(0.2)
        liveThread.join(0.2)

        self.assertFalse(registerThread.is_alive())
        self.assertFalse(liveThread.is_alive())
        self.assertEqual(workerOutcome, {})
        turn_queue = router._turn_notifications["turn-interleaved"]
        self.assertEqual(turn_queue.qsize(), 2)
        self.assertIs(
            router.next_turn_notification("turn-interleaved"), item_notification
        )
        self.assertIs(router.next_turn_notification("turn-interleaved"), completion)
        router.register_turn("turn-interleaved")
        self.assertIs(router._turn_notifications["turn-interleaved"], turn_queue)

    def test_expired_bounded_attempt_does_not_start_or_poison(self) -> None:
        lifecycle = llm._ClientLifecycle()
        attempt = MagicMock(return_value=1)
        poison_runtime = MagicMock()

        with lifecycle.request_lease("Codex"):
            with self.assertRaisesRegex(llm.LLMTimeoutError, "timed out"):
                llm.run_codex_bounded_attempt(
                    attempt,
                    deadline=time.monotonic() - 1,
                    lifecycle=lifecycle,
                    poison_runtime=poison_runtime,
                )

        attempt.assert_not_called()
        poison_runtime.assert_not_called()
        self.assertFalse(lifecycle.poison_event.is_set())
        with lifecycle.request_lease("Codex"):
            result = llm.run_codex_bounded_attempt(
                lambda _register: 2,
                deadline=time.monotonic() + 1,
                lifecycle=lifecycle,
                poison_runtime=poison_runtime,
            )
        self.assertEqual(result, 2)
        poison_runtime.assert_not_called()

    def test_production_codex_client_declines_all_server_requests(self) -> None:
        client = llm._build_codex_client(llm.CodexConfig())

        self.assertIsInstance(client._router, llm._EarlyCompletionMessageRouter)
        for method in (
            "item/commandExecution/requestApproval",
            "item/fileChange/requestApproval",
            "server/unknownRequest",
        ):
            with self.subTest(method=method):
                self.assertEqual(
                    client._handle_server_request(
                        {"id": "request-1", "method": method, "params": {}}
                    ),
                    {"decision": "decline"},
                )


class CodexClientTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        base = Path(self.tempdir.name)
        self.project = base / "project"
        self.home = base / "codex-home"
        self.project.mkdir()
        self.home.mkdir(mode=0o700)
        if os.name == "posix":
            self.home.chmod(0o700)
        self.settings = Settings.model_construct(
            project_root=self.project,
            llm=LLMConfig(codex_home=self.home, timeout_seconds=10),
            opencode_go_api_key=None,
        )
        self.runtime = MagicMock()
        self.runtime.account.return_value = SimpleNamespace(
            account=SimpleNamespace(root=SimpleNamespace(type="chatgpt"))
        )
        self.runtime.models.return_value = SimpleNamespace(
            data=[SimpleNamespace(model="gpt-5.6-luna", id="luna")]
        )
        effectiveConfig: dict[str, object] = {
            "mcp_servers": {},
            "model_providers": {},
            "profile": None,
            "openai_base_url": None,
            "chatgpt_base_url": None,
            "model_catalog_json": None,
            "oss_provider": None,
        }
        for dottedPath, expectedValue in llm.CODEX_EFFECTIVE_CONFIG_REQUIREMENTS.items():
            target = effectiveConfig
            components = dottedPath.split(".")
            for component in components[:-1]:
                child = target.setdefault(component, {})
                assert isinstance(child, dict)
                target = child
            target[components[-1]] = expectedValue
        effectiveModel = MagicMock()
        effectiveModel.model_dump.return_value = effectiveConfig
        self.runtime._client.request.return_value = SimpleNamespace(
            config=effectiveModel
        )

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def make_client(self) -> llm.CodexStructuredClient:
        with patch.object(llm, "Codex", return_value=self.runtime):
            return llm.CodexStructuredClient(self.settings)

    def set_turn_results(self, *results: object) -> tuple[MagicMock, list[MagicMock]]:
        thread = MagicMock()
        handles: list[MagicMock] = []
        for result in results:
            handle = MagicMock()
            if isinstance(result, BaseException):
                handle.run.side_effect = result
            else:
                handle.run.return_value = result
            handles.append(handle)
        thread.turn.side_effect = handles
        self.runtime.thread_start.return_value = thread
        return thread, handles

    def test_security_config_preflight_and_request_arguments(self) -> None:
        client = self.make_client()
        with patch.object(llm, "Codex", return_value=self.runtime) as codex_factory:
            second_client = llm.CodexStructuredClient(self.settings)
        config = codex_factory.call_args.kwargs["config"]
        self.assertEqual(config.config_overrides, llm.CODEX_CONFIG_OVERRIDES)
        expected_overrides = {
            'forced_login_method="chatgpt"',
            'model_provider="openai"',
            'web_search="disabled"',
            'history.persistence="none"',
            "analytics.enabled=false",
            "feedback.enabled=false",
            "check_for_update_on_startup=false",
            'otel.exporter="none"',
            'otel.trace_exporter="none"',
            'otel.metrics_exporter="none"',
            'shell_environment_policy.inherit="none"',
            "features.shell_tool=false",
            "features.unified_exec=false",
            "features.code_mode=false",
            "features.code_mode_host=false",
            "features.apps=false",
            "features.hooks=false",
            "features.memories=false",
            "features.multi_agent=false",
            "features.multi_agent_v2=false",
            "features.plugins=false",
            "features.remote_plugin=false",
            "skills.bundled.enabled=false",
            "orchestrator.skills.enabled=false",
            "orchestrator.mcp.enabled=false",
            "agents.enabled=false",
            "mcp_servers={}",
        }
        self.assertTrue(expected_overrides.issubset(config.config_overrides))
        self.assertEqual(config.env["CODEX_HOME"], str(self.home.resolve()))
        for name in ("OPENAI_API_KEY", "CODEX_API_KEY", "OPENCODE_GO_API_KEY"):
            self.assertEqual(config.env[name], "")
        self.assertFalse(Path(config.cwd).resolve().is_relative_to(self.project.resolve()))
        launch_args = config.launch_args_override
        self.assertIsNotNone(launch_args)
        assert launch_args is not None
        self.assertEqual(launch_args[:2], ("/usr/bin/env", "-i"))
        self.assertIn(f"CODEX_HOME={self.home.resolve()}", launch_args)
        self.assertIn(f"TMPDIR={Path(config.cwd).resolve()}", launch_args)
        self.assertIn("LANG=C", launch_args)
        self.assertIn("LC_ALL=C", launch_args)
        self.assertEqual(launch_args[-3:], ("app-server", "--listen", "stdio://"))
        for override in llm.CODEX_CONFIG_OVERRIDES:
            self.assertIn(("--config", override), tuple(zip(launch_args, launch_args[1:])))
        with patch.dict(os.environ, {"FRESHLIT_TEST_SECRET": "do-not-inherit"}):
            isolated_config = llm.build_codex_config(
                self.home.resolve(), Path(config.cwd).resolve()
            )
        self.assertNotIn(
            "do-not-inherit", "\0".join(isolated_config.launch_args_override or ())
        )
        self.assertNotIn("FRESHLIT_TEST_SECRET", isolated_config.env or {})
        self.runtime.account.assert_called()
        self.runtime.models.assert_called()
        self.runtime.login_api_key.assert_not_called()
        self.runtime._client.request.assert_called_with(
            "config/read",
            {"includeLayers": False},
            response_model=llm.ConfigReadResponse,
        )

        thread, _ = self.set_turn_results(
            completed_result('{"value":7,"nested":{"label":"ok"}}')
        )
        result = client.create_structured(
            self.settings,
            OutputModel,
            [
                {"role": "system", "content": "Caller policy"},
                {"role": "user", "content": "Untrusted paper"},
            ],
        )
        self.assertEqual(result.value, 7)
        start_args = self.runtime.thread_start.call_args.kwargs
        self.assertEqual(start_args["approval_mode"], ApprovalMode.deny_all)
        self.assertEqual(start_args["sandbox"], Sandbox.read_only)
        self.assertEqual(start_args["model_provider"], "openai")
        self.assertTrue(start_args["ephemeral"])
        self.assertEqual(start_args["developer_instructions"], "Caller policy")
        request_cwd = Path(start_args["cwd"])
        self.assertFalse(request_cwd.is_relative_to(self.project))
        turn_args = thread.turn.call_args
        self.assertEqual(turn_args.args[0], "Untrusted paper")
        self.assertFalse(turn_args.kwargs["output_schema"]["additionalProperties"])
        second_client.close()
        client.close()

    def test_account_and_model_preflight_fail_closed(self) -> None:
        self.runtime.account.return_value = SimpleNamespace(
            account=SimpleNamespace(root=SimpleNamespace(type="apiKey"))
        )
        with patch.object(llm, "Codex", return_value=self.runtime):
            with self.assertRaisesRegex(RuntimeError, "ChatGPT"):
                llm.CodexStructuredClient(self.settings)
        self.runtime.close.assert_called_once()

        self.runtime.close.reset_mock()
        self.runtime.account.return_value = SimpleNamespace(
            account=SimpleNamespace(root=SimpleNamespace(type="chatgpt"))
        )
        self.runtime.models.return_value = SimpleNamespace(data=[])
        with patch.object(llm, "Codex", return_value=self.runtime):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                llm.CodexStructuredClient(self.settings)
        self.runtime.close.assert_called_once()

    def test_effective_config_preflight_fails_closed(self) -> None:
        effectiveModel = MagicMock()
        effectiveModel.model_dump.return_value = {
            "model_provider": "custom",
            "mcp_servers": {},
            "model_providers": {},
        }
        self.runtime._client.request.return_value = SimpleNamespace(
            config=effectiveModel
        )
        with patch.object(llm, "Codex", return_value=self.runtime):
            with self.assertRaisesRegex(RuntimeError, "safety configuration"):
                llm.CodexStructuredClient(self.settings)
        self.runtime.account.assert_not_called()
        self.runtime.models.assert_not_called()
        self.runtime.close.assert_called_once()

    def test_startup_deadline_covers_constructor_and_retains_child(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        closeRelease = threading.Event()
        process = FakeProcess()

        class BlockingCodex:
            def __init__(self, *, config: object) -> None:
                self.config = config
                self._client = SimpleNamespace(_process=process)
                entered.set()
                release.wait(1)

            def close(self) -> None:
                closeRelease.wait(1)

        config = MagicMock()
        started = time.monotonic()
        try:
            with (
                patch.object(llm, "Codex", BlockingCodex),
                patch.object(llm, "CODEX_STARTUP_CAP_SECONDS", 0.02),
                patch.object(llm, "CLIENT_CLOSE_GRACE_SECONDS", 0.01),
                patch.object(llm, "PROCESS_TERMINATE_GRACE_SECONDS", 0.01),
                patch.object(llm, "PROCESS_KILL_GRACE_SECONDS", 0.01),
            ):
                with self.assertRaisesRegex(RuntimeError, "startup timed out"):
                    llm.start_codex_runtime(config, "gpt-5.6-luna", 1.0)
            self.assertTrue(entered.is_set())
            self.assertLess(time.monotonic() - started, 0.15)
            self.assertEqual(process.terminateCalls, 1)
            self.assertEqual(process.killCalls, 1)
            self.assertEqual(process.poll(), -9)
        finally:
            release.set()
            closeRelease.set()

    def test_startup_deadline_covers_blocked_preflight(self) -> None:
        entered = threading.Event()
        release = threading.Event()

        def block_config(*_args: object, **_kwargs: object) -> object:
            entered.set()
            release.wait(1)
            return object()

        self.runtime._client.request.side_effect = block_config
        started = time.monotonic()
        try:
            with (
                patch.object(llm, "Codex", return_value=self.runtime),
                patch.object(llm, "CODEX_STARTUP_CAP_SECONDS", 0.02),
                patch.object(llm, "CLIENT_CLOSE_GRACE_SECONDS", 0.01),
            ):
                with self.assertRaisesRegex(RuntimeError, "startup timed out"):
                    llm.start_codex_runtime(MagicMock(), "gpt-5.6-luna", 1.0)
            self.assertTrue(entered.is_set())
            self.assertLess(time.monotonic() - started, 0.1)
            self.runtime.close.assert_called_once()
        finally:
            release.set()

    @unittest.skipUnless(os.name == "posix", "POSIX permission check")
    def test_codex_home_rejects_group_or_other_permissions(self) -> None:
        self.home.chmod(0o750)
        with patch.object(llm, "Codex") as codex_factory:
            with self.assertRaisesRegex(RuntimeError, "permissions"):
                llm.CodexStructuredClient(self.settings)
        codex_factory.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "POSIX permission check")
    def test_codex_home_requires_exact_0700_permissions(self) -> None:
        self.home.chmod(0o600)
        try:
            with patch.object(llm, "Codex") as codex_factory:
                with self.assertRaisesRegex(RuntimeError, "0700"):
                    llm.CodexStructuredClient(self.settings)
            codex_factory.assert_not_called()
        finally:
            self.home.chmod(0o700)

    @unittest.skipUnless(os.name == "posix", "POSIX ownership check")
    def test_codex_home_rejects_other_owner(self) -> None:
        with (
            patch.object(llm.os, "getuid", return_value=self.home.stat().st_uid + 1),
            patch.object(llm, "Codex") as codex_factory,
        ):
            with self.assertRaisesRegex(RuntimeError, "owned"):
                llm.CodexStructuredClient(self.settings)
        codex_factory.assert_not_called()

    @unittest.skipUnless(os.name == "posix", "POSIX dedicated-profile check")
    def test_codex_home_rejects_shared_standard_profile(self) -> None:
        base = self.project.parent
        shared_home = base / ".codex"
        shared_home.mkdir(mode=0o700)
        shared_home.chmod(0o700)
        self.settings.llm.codex_home = shared_home
        with (
            patch.object(llm.Path, "home", return_value=base),
            patch.object(llm, "Codex") as codex_factory,
        ):
            with self.assertRaisesRegex(RuntimeError, "shared"):
                llm.CodexStructuredClient(self.settings)
        codex_factory.assert_not_called()

    def test_secure_launcher_unavailable_fails_closed(self) -> None:
        with (
            patch.object(llm, "CODEX_ENV_LAUNCHER", Path("/missing/env")),
            patch.object(llm, "Codex") as codex_factory,
        ):
            with self.assertRaisesRegex(RuntimeError, "launcher"):
                llm.CodexStructuredClient(self.settings)
        codex_factory.assert_not_called()

    def test_codex_home_must_be_explicitly_configured(self) -> None:
        self.settings.llm.codex_home = None
        with patch.object(llm, "Codex") as codex_factory:
            with self.assertRaisesRegex(RuntimeError, "FRESHLIT_CODEX_HOME"):
                llm.CodexStructuredClient(self.settings)
        codex_factory.assert_not_called()

    def test_output_validation_retries_only_up_to_bound(self) -> None:
        self.settings.llm.max_retries = 1
        thread, _ = self.set_turn_results(
            completed_result('{"value":"bad","nested":{"label":"x"}}'),
            completed_result('{"value":9,"nested":{"label":"ok"}}'),
        )
        client = self.make_client()
        result = client.create_structured(
            self.settings,
            OutputModel,
            [{"role": "user", "content": "fixture"}],
        )
        self.assertEqual(result.value, 9)
        self.assertEqual(thread.turn.call_count, 2)
        client.close()

        self.set_turn_results(
            completed_result('{"value":"bad","nested":{"label":"x"}}'),
            completed_result('{"value":"bad","nested":{"label":"x"}}'),
        )
        client = self.make_client()
        with self.assertRaises(llm.LLMOutputError) as caught:
            client.create_structured(
                self.settings,
                OutputModel,
                [{"role": "user", "content": "fixture"}],
            )
        self.assertNotIsInstance(caught.exception, llm.LLMNoFallbackError)
        client.close()

    def test_retryable_overload_retries_but_other_errors_do_not(self) -> None:
        self.settings.llm.max_retries = 1
        client = self.make_client()
        thread, _ = self.set_turn_results(
            completed_result('{"value":3,"nested":{"label":"ok"}}')
        )
        self.runtime.thread_start.side_effect = [
            ServerBusyError(-32000, "busy", "server_overloaded"),
            thread,
        ]
        result = client.create_structured(
            self.settings,
            OutputModel,
            [{"role": "user", "content": "fixture"}],
        )
        self.assertEqual(result.value, 3)
        self.assertEqual(self.runtime.thread_start.call_count, 2)

        self.runtime.thread_start.reset_mock()
        self.runtime.thread_start.side_effect = ValueError("unsafe detail")
        with self.assertRaisesRegex(llm.LLMRequestError, "request failed") as caught:
            client.create_structured(
                self.settings,
                OutputModel,
                [{"role": "user", "content": "secret prompt"}],
            )
        self.assertNotIn("unsafe detail", str(caught.exception))
        self.assertEqual(self.runtime.thread_start.call_count, 1)
        client.close()

    def _assert_blocking_stage_times_out(
        self, stage: str, *, hanging_interrupt: bool = False
    ) -> None:
        self.settings.llm.timeout_seconds = 0.025
        release = threading.Event()
        entered = threading.Event()
        interrupt_entered = threading.Event()
        interrupt_release = threading.Event()
        close_release = threading.Event()
        thread = MagicMock()
        handle = MagicMock()

        if stage == "thread_start":
            def block_start(**_kwargs: object) -> MagicMock:
                entered.set()
                release.wait(1)
                return thread

            self.runtime.thread_start.side_effect = block_start
        else:
            self.runtime.thread_start.return_value = thread
            if stage == "turn":
                def block_turn(*_args: object, **_kwargs: object) -> MagicMock:
                    entered.set()
                    release.wait(1)
                    return handle

                thread.turn.side_effect = block_turn
            else:
                thread.turn.return_value = handle

                def block_run() -> object:
                    entered.set()
                    release.wait(1)
                    return completed_result(
                        '{"value":1,"nested":{"label":"ok"}}'
                    )

                handle.run.side_effect = block_run

        if hanging_interrupt:
            def block_interrupt() -> None:
                interrupt_entered.set()
                interrupt_release.wait(1)

            handle.interrupt.side_effect = block_interrupt

            def block_close() -> None:
                close_release.wait(1)

            self.runtime.close.side_effect = block_close

        client = self.make_client()
        started = time.monotonic()
        try:
            with (
                patch.object(llm, "CODEX_CANCELLATION_GRACE_SECONDS", 0.01),
                patch.object(llm, "CLIENT_CLOSE_GRACE_SECONDS", 0.01),
            ):
                with self.assertRaisesRegex(llm.LLMTimeoutError, "timed out"):
                    client.create_structured(
                        self.settings,
                        OutputModel,
                        [{"role": "user", "content": "fixture"}],
                    )
            self.assertTrue(entered.is_set())
            self.assertLess(time.monotonic() - started, 0.2)
            self.runtime.close.assert_called_once()
            if stage == "run":
                handle.interrupt.assert_called_once()
            if hanging_interrupt:
                self.assertTrue(interrupt_entered.wait(0.05))
            with self.assertRaises(llm.LLMNoFallbackError):
                client.create_structured(
                    self.settings,
                    OutputModel,
                    [{"role": "user", "content": "fixture"}],
                )
        finally:
            release.set()
            interrupt_release.set()
            close_release.set()
            time.sleep(0.01)
            client.close()

    def test_timeout_covers_blocking_thread_start(self) -> None:
        self._assert_blocking_stage_times_out("thread_start")

    def test_timeout_covers_blocking_thread_turn(self) -> None:
        self._assert_blocking_stage_times_out("turn")

    def test_timeout_covers_blocking_run_and_hanging_interrupt(self) -> None:
        self._assert_blocking_stage_times_out("run", hanging_interrupt=True)

    def test_completed_cancellation_grace_does_not_poison_runtime(self) -> None:
        self.settings.llm.timeout_seconds = 0.025
        entered = threading.Event()
        release = threading.Event()
        thread = MagicMock()
        first_handle = MagicMock()
        second_handle = MagicMock()

        def interrupted_run() -> object:
            entered.set()
            release.wait(1)
            return completed_result('{"value":1,"nested":{"label":"late"}}')

        first_handle.run.side_effect = interrupted_run
        first_handle.interrupt.side_effect = release.set
        second_handle.run.return_value = completed_result(
            '{"value":2,"nested":{"label":"healthy"}}'
        )
        thread.turn.side_effect = [first_handle, second_handle]
        self.runtime.thread_start.return_value = thread
        client = self.make_client()

        with patch.object(llm, "CODEX_CANCELLATION_GRACE_SECONDS", 0.1):
            with self.assertRaisesRegex(llm.LLMTimeoutError, "timed out"):
                client.create_structured(
                    self.settings,
                    OutputModel,
                    [{"role": "user", "content": "first"}],
                )

        self.assertTrue(entered.is_set())
        first_handle.interrupt.assert_called_once()
        self.assertFalse(client._lifecycle.poison_event.is_set())
        self.runtime.close.assert_not_called()

        self.settings.llm.timeout_seconds = 1.0
        result = client.create_structured(
            self.settings,
            OutputModel,
            [{"role": "user", "content": "second"}],
        )
        self.assertEqual(result.value, 2)
        self.assertEqual(thread.turn.call_count, 2)
        client.close()

    def test_inactive_deadlines_do_not_poison_or_block_next_request(self) -> None:
        self.settings.llm.timeout_seconds = 1.0
        self.settings.llm.max_retries = 0
        thread, _ = self.set_turn_results(
            completed_result('{"value":5,"nested":{"label":"healthy"}}')
        )
        client = self.make_client()

        with self.assertRaisesRegex(llm.LLMTimeoutError, "timed out"):
            client._run_attempt(
                self.settings,
                OutputModel,
                [{"role": "user", "content": "expired"}],
                time.monotonic() - 1,
            )
        self.runtime.thread_start.assert_not_called()

        with (
            patch.object(
                client, "_run_attempt", side_effect=ValueError("completed failure")
            ),
            patch.object(llm.time, "monotonic", side_effect=[0.0, 2.0]),
        ):
            with self.assertRaisesRegex(llm.LLMTimeoutError, "timed out"):
                client.create_structured(
                    self.settings,
                    OutputModel,
                    [{"role": "user", "content": "completed"}],
                )

        self.assertFalse(client._lifecycle.poison_event.is_set())
        self.runtime.close.assert_not_called()
        result = client.create_structured(
            self.settings,
            OutputModel,
            [{"role": "user", "content": "next"}],
        )
        self.assertEqual(result.value, 5)
        self.assertEqual(thread.turn.call_count, 1)
        client.close()

    def test_noncompleted_and_nonallowlisted_activity_are_no_fallback(self) -> None:
        self.settings.llm.max_retries = 2
        noncompleted = SimpleNamespace(
            status="interrupted", final_response="{}", items=[]
        )
        self.set_turn_results(noncompleted)
        client = self.make_client()
        with self.assertRaisesRegex(llm.LLMNoFallbackError, "did not complete"):
            client.create_structured(
                self.settings,
                OutputModel,
                [{"role": "user", "content": "fixture"}],
            )

        for item_type in ("imageView", "imageGeneration", "mcpToolCall", "newType"):
            self.set_turn_results(completed_result("{}", item_type))
            with self.subTest(item_type=item_type):
                with self.assertRaisesRegex(
                    llm.LLMUnsafeActivityError, "unsafe activity"
                ):
                    client.create_structured(
                        self.settings,
                        OutputModel,
                        [{"role": "user", "content": "fixture"}],
                    )

        passive_without_agent = SimpleNamespace(
            status="completed",
            final_response="{}",
            items=[SimpleNamespace(root=SimpleNamespace(type="reasoning"))],
        )
        self.set_turn_results(passive_without_agent)
        with self.assertRaisesRegex(llm.LLMUnsafeActivityError, "agent message"):
            client.create_structured(
                self.settings,
                OutputModel,
                [{"role": "user", "content": "fixture"}],
            )
        client.close()

    def test_passive_item_allowlist_accepts_all_proven_types(self) -> None:
        items = [
            SimpleNamespace(root=SimpleNamespace(type=item_type))
            for item_type in ("userMessage", "reasoning", "agentMessage")
        ]
        result = SimpleNamespace(
            status="completed",
            final_response='{"value":2,"nested":{"label":"ok"}}',
            items=items,
        )
        self.set_turn_results(result)
        client = self.make_client()
        parsed = client.create_structured(
            self.settings,
            OutputModel,
            [{"role": "user", "content": "fixture"}],
        )
        self.assertEqual(parsed.value, 2)
        client.close()

    def test_close_waits_for_accepted_request_and_rejects_new_request(self) -> None:
        entered = threading.Event()
        release = threading.Event()
        result_holder: list[OutputModel] = []
        thread, handles = self.set_turn_results(
            completed_result('{"value":4,"nested":{"label":"ok"}}')
        )

        def blocking_run() -> object:
            entered.set()
            release.wait(1)
            return completed_result('{"value":4,"nested":{"label":"ok"}}')

        handles[0].run.side_effect = blocking_run
        client = self.make_client()
        request_thread = threading.Thread(
            target=lambda: result_holder.append(
                client.create_structured(
                    self.settings,
                    OutputModel,
                    [{"role": "user", "content": "fixture"}],
                )
            )
        )
        request_thread.start()
        self.assertTrue(entered.wait(0.2))
        close_thread = threading.Thread(target=client.close)
        close_thread.start()
        deadline = time.monotonic() + 0.2
        while not client._lifecycle.closing and time.monotonic() < deadline:
            time.sleep(0.001)
        with self.assertRaisesRegex(llm.LLMRequestError, "closed"):
            client.create_structured(
                self.settings,
                OutputModel,
                [{"role": "user", "content": "fixture"}],
            )
        self.runtime.close.assert_not_called()
        release.set()
        request_thread.join(0.2)
        close_thread.join(0.2)
        self.assertEqual(result_holder[0].value, 4)
        self.assertEqual(thread.turn.call_count, 1)
        self.runtime.close.assert_called_once()

    def test_close_is_idempotent(self) -> None:
        client = self.make_client()
        client.close()
        client.close()
        self.runtime.close.assert_called_once()

    def test_hanging_graceful_close_terminates_then_kills_owned_child(self) -> None:
        process = FakeProcess()
        closeEntered = threading.Event()
        closeRelease = threading.Event()
        runtime = SimpleNamespace(
            _client=SimpleNamespace(_process=process),
        )

        def blocking_close() -> None:
            closeEntered.set()
            closeRelease.wait(1)

        runtime.close = blocking_close
        owner = llm.OwnedCodexRuntime(runtime)
        try:
            with (
                patch.object(llm, "PROCESS_TERMINATE_GRACE_SECONDS", 0.01),
                patch.object(llm, "PROCESS_KILL_GRACE_SECONDS", 0.01),
            ):
                self.assertTrue(owner.close(0.01))
            self.assertTrue(closeEntered.is_set())
            self.assertEqual(process.terminateCalls, 1)
            self.assertEqual(process.killCalls, 1)
            self.assertEqual(process.poll(), -9)
            self.assertTrue(owner.close(0.01))
            self.assertEqual(process.killCalls, 1)
        finally:
            closeRelease.set()

    def test_poison_wakes_concurrent_attempt_waiter_promptly(self) -> None:
        firstEntered = threading.Event()
        secondEntered = threading.Event()
        release = threading.Event()
        callLock = threading.Lock()
        callCount = 0

        def thread_start(**_kwargs: object) -> MagicMock:
            nonlocal callCount
            with callLock:
                callCount += 1
                current = callCount
            thread = MagicMock()
            handle = MagicMock()

            def blocking_run() -> object:
                (firstEntered if current == 1 else secondEntered).set()
                release.wait(1)
                return completed_result('{"value":1,"nested":{"label":"ok"}}')

            handle.run.side_effect = blocking_run
            thread.turn.return_value = handle
            return thread

        self.runtime.thread_start.side_effect = thread_start
        client = self.make_client()
        shortSettings = self.settings.model_copy(deep=True)
        shortSettings.llm.timeout_seconds = 0.04
        longSettings = self.settings.model_copy(deep=True)
        longSettings.llm.timeout_seconds = 1.0
        outcomes: dict[str, BaseException] = {}

        def request(name: str, settings: Settings) -> None:
            try:
                client.create_structured(
                    settings,
                    OutputModel,
                    [{"role": "user", "content": "fixture"}],
                )
            except BaseException as exc:
                outcomes[name] = exc

        first = threading.Thread(target=request, args=("first", shortSettings))
        second = threading.Thread(target=request, args=("second", longSettings))
        started = time.monotonic()
        try:
            with (
                patch.object(llm, "CODEX_CANCELLATION_GRACE_SECONDS", 0.005),
                patch.object(llm, "CLIENT_CLOSE_GRACE_SECONDS", 0.01),
            ):
                first.start()
                self.assertTrue(firstEntered.wait(0.1))
                second.start()
                self.assertTrue(secondEntered.wait(0.1))
                first.join(0.2)
                second.join(0.2)
            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
            self.assertLess(time.monotonic() - started, 0.25)
            self.assertIsInstance(outcomes["first"], llm.LLMTimeoutError)
            self.assertIsInstance(outcomes["second"], llm.LLMNoFallbackError)
            self.assertNotIsInstance(outcomes["second"], llm.LLMTimeoutError)
            self.assertTrue(client._lifecycle.poison_event.is_set())
        finally:
            release.set()
            first.join(0.2)
            second.join(0.2)
            client.close()


class DispatchAndGatewayTests(unittest.TestCase):
    def settings(self, provider: str, key: str | None = None) -> Settings:
        return Settings.model_construct(
            project_root=Path("/tmp/project"),
            llm=LLMConfig(provider=provider),
            opencode_go_api_key=key,
        )

    def test_dispatch_has_no_provider_fallback(self) -> None:
        settings = self.settings("codex")
        with (
            patch.object(
                llm.CodexStructuredClient,
                "__init__",
                side_effect=RuntimeError("codex failed"),
                autospec=True,
            ),
            patch.object(llm, "GatewayStructuredClient") as gateway,
        ):
            with self.assertRaisesRegex(RuntimeError, "codex failed"):
                llm.build_client(settings)
        gateway.assert_not_called()

        gateway_settings = self.settings("gateway", "key")
        with patch.object(llm, "GatewayStructuredClient") as gateway:
            llm.build_client(gateway_settings)
        gateway.assert_called_once_with(gateway_settings)

    def test_gateway_key_close_and_mode_are_per_instance(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "required"):
            llm.GatewayStructuredClient(self.settings("gateway"))

        raw_a = MagicMock()
        raw_b = MagicMock()
        with patch.object(llm, "OpenAI", side_effect=[raw_a, raw_b]):
            first = llm.GatewayStructuredClient(self.settings("gateway", "key-a"))
            second = llm.GatewayStructuredClient(self.settings("gateway", "key-b"))
        first._working_mode = llm.instructor.Mode.JSON
        self.assertIsNone(second._working_mode)
        first.close()
        first.close()
        second.close()
        raw_a.close.assert_called_once()
        raw_b.close.assert_called_once()

    def test_gateway_close_request_lease_is_atomic(self) -> None:
        raw = MagicMock()
        entered = threading.Event()
        release = threading.Event()
        structured = MagicMock()

        def blocking_create(**_kwargs: object) -> OutputModel:
            entered.set()
            release.wait(1)
            return OutputModel(value=6)

        structured.chat.completions.create.side_effect = blocking_create
        with (
            patch.object(llm, "OpenAI", return_value=raw),
            patch.object(llm.instructor, "from_openai", return_value=structured),
        ):
            client = llm.GatewayStructuredClient(self.settings("gateway", "key"))
            results: list[OutputModel] = []
            request_thread = threading.Thread(
                target=lambda: results.append(
                    client.create_structured(
                        self.settings("gateway", "key"),
                        OutputModel,
                        [{"role": "user", "content": "fixture"}],
                    )
                )
            )
            request_thread.start()
            self.assertTrue(entered.wait(0.2))
            close_thread = threading.Thread(target=client.close)
            close_thread.start()
            deadline = time.monotonic() + 0.2
            while not client._lifecycle.closing and time.monotonic() < deadline:
                time.sleep(0.001)
            with self.assertRaisesRegex(llm.LLMRequestError, "closed"):
                client.create_structured(
                    self.settings("gateway", "key"),
                    OutputModel,
                    [{"role": "user", "content": "fixture"}],
                )
            raw.close.assert_not_called()
            release.set()
            request_thread.join(0.2)
            close_thread.join(0.2)

        self.assertEqual(results[0].value, 6)
        raw.close.assert_called_once()

    def test_gateway_passes_explicit_parse_only_retry_policy(self) -> None:
        settings = self.settings("gateway", "key")
        settings.llm.max_retries = 2
        raw = MagicMock()
        structured = MagicMock()
        structured.chat.completions.create.return_value = OutputModel(value=3)
        with (
            patch.object(llm, "OpenAI", return_value=raw),
            patch.object(llm.instructor, "from_openai", return_value=structured),
        ):
            client = llm.GatewayStructuredClient(settings)
            result = client.create_structured(
                settings,
                OutputModel,
                [{"role": "user", "content": "fixture"}],
            )

        retry_policy = structured.chat.completions.create.call_args.kwargs[
            "max_retries"
        ]
        self.assertEqual(result.value, 3)
        self.assertIsInstance(retry_policy, llm.Retrying)
        self.assertTrue(retry_policy.reraise)
        self.assertEqual(retry_policy.stop.max_attempt_number, 3)
        self.assertEqual(
            retry_policy.retry.exception_types,
            (ValidationError, llm.json.JSONDecodeError),
        )

    def gateway_status_error(
        self, status_code: int, body: object | None = None
    ) -> APIStatusError:
        request = httpx.Request(
            "POST", "https://secret-gateway.invalid/private?key=secret-key"
        )
        response = httpx.Response(status_code, request=request)
        return APIStatusError(
            "raw provider error with secret body",
            response=response,
            body=(
                {"prompt": "secret prompt", "key": "secret-key"}
                if body is None
                else body
            ),
        )

    def wrapped_error(self, inner: Exception) -> Exception:
        try:
            raise inner
        except Exception as caught:
            try:
                raise RuntimeError("raw instructor wrapper detail") from caught
            except RuntimeError as wrapped:
                return wrapped

    def assert_gateway_stops_after_one_call(self, error: Exception) -> None:
        raw = MagicMock()
        structured = MagicMock()
        structured.chat.completions.create.side_effect = error
        with (
            patch.object(llm, "OpenAI", return_value=raw),
            patch.object(
                llm.instructor, "from_openai", return_value=structured
            ) as from_openai,
            patch.object(llm.log, "debug") as debug_log,
        ):
            client = llm.GatewayStructuredClient(
                self.settings("gateway", "secret-key")
            )
            with self.assertRaises(llm.LLMNoFallbackError) as caught:
                client.create_structured(
                    self.settings("gateway", "secret-key"),
                    OutputModel,
                    [{"role": "user", "content": "secret prompt"}],
                )

        self.assertEqual(structured.chat.completions.create.call_count, 1)
        self.assertEqual(from_openai.call_count, 1)
        self.assertEqual(str(caught.exception), "Gateway structured request failed")
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        exposed = f"{caught.exception}\n{debug_log.call_args_list}"
        for secret in (
            "raw provider error",
            "raw instructor wrapper",
            "secret prompt",
            "secret-key",
            "secret-gateway.invalid",
        ):
            self.assertNotIn(secret, exposed)

    def test_gateway_terminal_status_errors_stop_after_one_call(self) -> None:
        for status_code in (401, 403, 429, 500, 503):
            with self.subTest(status_code=status_code):
                self.assert_gateway_stops_after_one_call(
                    self.wrapped_error(self.gateway_status_error(status_code))
                )

    def test_gateway_generic_compatible_statuses_stop_after_one_call(self) -> None:
        errors = [
            self.gateway_status_error(status_code)
            for status_code in (400, 404, 422)
        ]
        errors.extend(
            [
                self.gateway_status_error(
                    404,
                    {
                        "error": {
                            "code": "model_not_found",
                            "param": "model",
                            "message": "response_format is unsupported",
                        },
                        "request": {
                            "tools": [{"type": "function"}],
                            "response_format": {"type": "json_schema"},
                        },
                    },
                ),
                self.gateway_status_error(
                    422,
                    {
                        "error": {
                            "code": "context_length_exceeded",
                            "param": "messages",
                        },
                        "request": {
                            "tools": [{"type": "function"}],
                            "response_format": {"type": "json_schema"},
                        },
                    },
                ),
                self.gateway_status_error(
                    400,
                    {
                        "error": {
                            "code": "content_policy_violation",
                            "param": "prompt",
                        }
                    },
                ),
                self.gateway_status_error(
                    404, {"error": {"code": "endpoint_not_found"}}
                ),
            ]
        )
        for error in errors:
            with self.subTest(body=error.body):
                self.assert_gateway_stops_after_one_call(self.wrapped_error(error))

    def test_gateway_transport_and_unknown_errors_stop_after_one_call(self) -> None:
        timeout = APITimeoutError(
            httpx.Request("POST", "https://secret-gateway.invalid/private")
        )
        connection = APIConnectionError(
            message="raw connection detail",
            request=httpx.Request(
                "POST", "https://secret-gateway.invalid/private"
            ),
        )
        for error in (
            self.wrapped_error(timeout),
            self.wrapped_error(connection),
            RuntimeError("closed transport raw detail"),
            RuntimeError("unknown raw provider failure"),
        ):
            with self.subTest(error_type=type(error).__name__):
                self.assert_gateway_stops_after_one_call(error)

    def test_real_instructor_terminal_statuses_make_one_http_attempt(self) -> None:
        cases = (
            (401, {"error": {"code": "invalid_api_key", "message": "secret"}}),
            (400, {"error": {"code": "bad_request", "message": "secret"}}),
            (404, {"error": {"code": "endpoint_not_found", "message": "secret"}}),
            (
                400,
                {
                    "error": {
                        "code": "context_length_exceeded",
                        "param": "messages",
                        "message": "secret",
                    }
                },
            ),
            (429, {"error": {"code": "rate_limit_exceeded", "message": "secret"}}),
            (503, {"error": {"code": "server_error", "message": "secret"}}),
        )
        for status_code, body in cases:
            with self.subTest(status_code=status_code, code=body["error"]["code"]):
                secretMarker = f"provider-secret-{status_code}-{body['error']['code']}"
                body["error"]["message"] = secretMarker
                requestCount = 0

                def respond(request: httpx.Request) -> httpx.Response:
                    nonlocal requestCount
                    requestCount += 1
                    return httpx.Response(status_code, request=request, json=body)

                transport = httpx.MockTransport(respond)

                def openai_factory(**kwargs: object) -> OpenAI:
                    return OpenAI(
                        **kwargs,
                        http_client=httpx.Client(transport=transport),
                    )

                for logger_name in (
                    "instructor",
                    *llm.INSTRUCTOR_RETRY_LOGGER_NAMES,
                ):
                    retry_logger = logging.getLogger(logger_name)
                    retry_logger.disabled = False
                    retry_logger.setLevel(logging.DEBUG)

                root_logger = logging.getLogger()
                root_level = root_logger.level
                root_disabled = root_logger.disabled
                log_stream = io.StringIO()
                stderr_stream = io.StringIO()
                capture_handler = logging.StreamHandler(log_stream)
                root_logger.addHandler(capture_handler)
                with patch.object(llm, "OpenAI", side_effect=openai_factory):
                    client = llm.GatewayStructuredClient(
                        self.settings("gateway", "test-key")
                    )
                try:
                    with redirect_stderr(stderr_stream):
                        logging.getLogger("freshlit.test.user").warning(
                            "user-root-logging-still-enabled"
                        )
                        with self.assertRaises(llm.LLMNoFallbackError) as caught:
                            client.create_structured(
                                self.settings("gateway", "test-key"),
                                OutputModel,
                                [{"role": "user", "content": "fixture"}],
                            )
                finally:
                    client.close()
                    root_logger.removeHandler(capture_handler)

                self.assertEqual(requestCount, 1)
                self.assertEqual(
                    str(caught.exception), "Gateway structured request failed"
                )
                self.assertIsNone(caught.exception.__cause__)
                self.assertIsNone(caught.exception.__context__)
                emitted = (
                    f"{caught.exception}\n{log_stream.getvalue()}\n"
                    f"{stderr_stream.getvalue()}"
                )
                self.assertNotIn(secretMarker, emitted)
                self.assertIn("user-root-logging-still-enabled", log_stream.getvalue())
                self.assertEqual(root_logger.level, root_level)
                self.assertEqual(root_logger.disabled, root_disabled)
                self.assertTrue(logging.getLogger("instructor").disabled)
                for logger_name in llm.INSTRUCTOR_RETRY_LOGGER_NAMES:
                    self.assertTrue(logging.getLogger(logger_name).disabled)

    def test_gateway_wrapped_400_negotiates_and_caches_successful_mode(self) -> None:
        raw = MagicMock()
        structured = MagicMock()
        structured.chat.completions.create.side_effect = [
            self.wrapped_error(
                self.gateway_status_error(
                    400,
                    {
                        "error": {
                            "code": "unsupported_response_format",
                            "param": "response_format",
                        }
                    },
                )
            ),
            OutputModel(value=8),
            OutputModel(value=9),
        ]
        with (
            patch.object(llm, "OpenAI", return_value=raw),
            patch.object(
                llm.instructor, "from_openai", return_value=structured
            ) as from_openai,
        ):
            client = llm.GatewayStructuredClient(self.settings("gateway", "key"))
            result = client.create_structured(
                self.settings("gateway", "key"),
                OutputModel,
                [{"role": "user", "content": "fixture"}],
            )
            cached_result = client.create_structured(
                self.settings("gateway", "key"),
                OutputModel,
                [{"role": "user", "content": "fixture"}],
            )

        self.assertEqual(result.value, 8)
        self.assertEqual(cached_result.value, 9)
        self.assertEqual(
            [call.kwargs["mode"] for call in from_openai.call_args_list],
            [
                llm.instructor.Mode.JSON,
                llm.instructor.Mode.TOOLS,
                llm.instructor.Mode.TOOLS,
            ],
        )
        self.assertEqual(client._working_mode, llm.instructor.Mode.TOOLS)

    def test_gateway_local_validation_error_can_negotiate(self) -> None:
        validation_error: ValidationError
        try:
            OutputModel.model_validate({"value": "invalid"})
        except ValidationError as caught:
            validation_error = caught

        raw = MagicMock()
        structured = MagicMock()
        structured.chat.completions.create.side_effect = [
            validation_error,
            OutputModel(value=10),
        ]
        with (
            patch.object(llm, "OpenAI", return_value=raw),
            patch.object(llm.instructor, "from_openai", return_value=structured),
        ):
            client = llm.GatewayStructuredClient(self.settings("gateway", "key"))
            result = client.create_structured(
                self.settings("gateway", "key"),
                OutputModel,
                [{"role": "user", "content": "fixture"}],
            )

        self.assertEqual(result.value, 10)
        self.assertEqual(structured.chat.completions.create.call_count, 2)

    def test_gateway_all_compatible_modes_raise_sanitized_output_error(self) -> None:
        raw = MagicMock()
        structured = MagicMock()
        structured.chat.completions.create.side_effect = [
            self.wrapped_error(self.gateway_status_error(status_code, body))
            for status_code, body in (
                (400, {"param": "response_format"}),
                (404, {"error": {"code": "tools_not_supported"}}),
                (422, {"error": {"field": "function_call"}}),
            )
        ]
        with (
            patch.object(llm, "OpenAI", return_value=raw),
            patch.object(llm.instructor, "from_openai", return_value=structured),
        ):
            client = llm.GatewayStructuredClient(
                self.settings("gateway", "secret-key")
            )
            with self.assertLogs(llm.log, level="DEBUG") as captured:
                with self.assertRaises(llm.LLMOutputError) as caught:
                    client.create_structured(
                        self.settings("gateway", "secret-key"),
                        OutputModel,
                        [{"role": "user", "content": "secret prompt"}],
                    )

        self.assertEqual(structured.chat.completions.create.call_count, 3)
        self.assertEqual(
            str(caught.exception),
            "Gateway does not support the requested structured output",
        )
        self.assertIsNone(caught.exception.__cause__)
        self.assertIsNone(caught.exception.__context__)
        exposed = f"{caught.exception}\n{' '.join(captured.output)}"
        for secret in (
            "raw provider error",
            "raw instructor wrapper",
            "secret prompt",
            "secret-key",
            "secret-gateway.invalid",
        ):
            self.assertNotIn(secret, exposed)


if __name__ == "__main__":
    unittest.main()
