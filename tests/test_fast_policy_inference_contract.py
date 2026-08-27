"""Contract test for the deployed Fast Policy inference path.

`AGENTS.md` section 11 asks every stage for an exported-inference-graph check, and
`docs/iteration-plan.md` section 8.5 makes it a precondition for promoting this
lineage to the default Fast Policy. This project's Fast Policy is not shipped as
an exported graph: it is served over websockets by
``deployment/model_server/server_policy.py``, and every eval client -- including
the one that produced the I4.5-R1 numbers -- reaches it through
``WebsocketClientPolicy``. The deployed inference surface is therefore the wire
contract, and that is what this test pins:

    handshake metadata -> request envelope -> ndarray codec -> response envelope
    -> PolicyServerWrapper un-normalization route

The model is stubbed, so no GPU and no checkpoint are needed; the websocket
transport, the msgpack-numpy codec and the wrapper's un-normalization route run
for real. The specific fields asserted here are the ones
``examples/simBenchmarks/LIBERO/eval_files/model2libero_interface.py`` reads:
``metadata["action_chunk_size"]`` and ``response["data"]["actions"]`` shaped
``(B, T, D)`` with the first seven dims carrying world_vector / rotation_delta /
gripper. Changing any of them silently breaks every eval client, which is the
failure mode an exported-graph check exists to catch.
"""

from __future__ import annotations

import socket
import threading
import unittest
from typing import Any, Dict, List, Optional
from unittest import mock

import numpy as np

from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer


BATCH = 1
CHUNK = 7
ACTION_DIM = 7
IMAGE_HW = (224, 224)


def _free_port() -> int:
    """Reserve a port, then release it so the server can bind it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _deterministic_chunk() -> np.ndarray:
    """(B, T, D) where element [b, t, d] == b * 100 + t * 10 + d.

    Encoding the index into the value means a transposed, truncated or
    re-ordered chunk cannot pass silently.
    """
    b = np.arange(BATCH, dtype=np.float32)[:, None, None] * 100.0
    t = np.arange(CHUNK, dtype=np.float32)[None, :, None] * 10.0
    d = np.arange(ACTION_DIM, dtype=np.float32)[None, None, :]
    return (b + t + d).astype(np.float32)


def _libero_style_request(unnorm_key: Optional[str] = "libero_all") -> Dict[str, Any]:
    """The exact request shape the LIBERO eval client sends (flat, no ``type``)."""
    image = np.zeros((*IMAGE_HW, 3), dtype=np.uint8)
    image[0, 0, 0] = 42  # marker: proves the codec did not rewrite pixels
    return {
        "examples": [
            {
                "image": [image],
                "lang": "pick up the black bowl",
                "state": np.arange(8, dtype=np.float32)[None, :],
            }
        ],
        "unnorm_key": unnorm_key,
        "do_sample": False,
        "use_ddim": True,
        "num_ddim_steps": 10,
    }


class _StubPolicy:
    """Duck-typed PolicyServerWrapper: records requests, returns a fixed chunk."""

    def __init__(self) -> None:
        self.requests: List[Dict[str, Any]] = []
        self.raise_on_next = False

    def predict_action(self, examples, unnorm_key=None, **kwargs) -> Dict[str, np.ndarray]:
        if self.raise_on_next:
            self.raise_on_next = False
            raise RuntimeError("stub inference failure")
        self.requests.append({"examples": examples, "unnorm_key": unnorm_key, "kwargs": kwargs})
        return {"actions": _deterministic_chunk()}


SERVER_METADATA = {
    "env": "starvla_policy_server",
    "ckpt_path": "stub://steps_6000_pytorch_model.pt",
    "action_chunk_size": CHUNK,
    "available_unnorm_keys": ["libero_all"],
    "default_unnorm_key": "libero_all",
    "training_data_mix": "libero_all",
    "training_obs_image_size": [IMAGE_HW[0], IMAGE_HW[1]],
    "action_keys": ["action.delta_eef"],
    "state_keys": ["state.eef"],
}


class FastPolicyWireContractTest(unittest.TestCase):
    """Real client, real server, real codec; only the model is stubbed."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = _StubPolicy()
        cls.port = _free_port()
        cls.server = WebsocketPolicyServer(
            policy=cls.policy,
            host="127.0.0.1",
            port=cls.port,
            idle_timeout=-1,
            metadata=SERVER_METADATA,
        )
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.client = WebsocketClientPolicy("127.0.0.1", cls.port)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.client.close()

    def test_handshake_metadata_survives_and_keeps_client_required_fields(self):
        meta = self.client.get_server_metadata()
        self.assertEqual(meta, SERVER_METADATA)
        # The LIBERO client does `int(meta["action_chunk_size"])` at construction:
        # a missing or renamed key is a hard failure, not a degradation.
        self.assertEqual(int(meta["action_chunk_size"]), CHUNK)
        self.assertEqual(meta["training_obs_image_size"], [IMAGE_HW[0], IMAGE_HW[1]])

    def test_infer_response_envelope_and_action_array(self):
        response = self.client.predict_action(_libero_style_request())

        self.assertEqual(response["status"], "ok")
        self.assertTrue(response["ok"])
        self.assertEqual(response["type"], "inference_result")
        self.assertEqual(response["request_id"], "default")

        actions = response["data"]["actions"]
        self.assertIsInstance(actions, np.ndarray)
        self.assertEqual(actions.shape, (BATCH, CHUNK, ACTION_DIM))
        self.assertEqual(actions.dtype, np.float32)
        np.testing.assert_array_equal(actions, _deterministic_chunk())

        # The client consumes actions[0] -> (T, D) and slices 0:3 / 3:6 / 6:7.
        per_step = actions[0]
        self.assertEqual(per_step.shape, (CHUNK, ACTION_DIM))
        self.assertGreaterEqual(per_step.shape[1], 7)
        np.testing.assert_array_equal(per_step[0, :3], [0.0, 1.0, 2.0])
        np.testing.assert_array_equal(per_step[0, 3:6], [3.0, 4.0, 5.0])
        np.testing.assert_array_equal(per_step[0, 6:7], [6.0])

    def test_request_side_reaches_the_policy_unchanged(self):
        request = _libero_style_request()
        self.client.predict_action(request)

        received = self.policy.requests[-1]
        self.assertEqual(received["unnorm_key"], "libero_all")
        # Inference kwargs must be forwarded, not swallowed: dropping use_ddim
        # would silently change the sampler and therefore the latency profile.
        self.assertEqual(
            received["kwargs"],
            {"do_sample": False, "use_ddim": True, "num_ddim_steps": 10},
        )

        example = received["examples"][0]
        self.assertEqual(example["lang"], "pick up the black bowl")
        self.assertEqual(len(example["image"]), 1)
        image = np.asarray(example["image"][0])
        self.assertEqual(image.shape, (*IMAGE_HW, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual(int(image[0, 0, 0]), 42)
        state = np.asarray(example["state"])
        self.assertEqual(state.shape, (1, 8))
        self.assertEqual(state.dtype, np.float32)

    def test_ping_and_unknown_type_do_not_wedge_the_connection(self):
        pong = self.client.predict_action({"type": "ping", "request_id": "p1"})
        self.assertEqual(pong, {"status": "ok", "ok": True, "type": "ping", "request_id": "p1"})

        unknown = self.client.predict_action({"type": "teleport", "request_id": "u1"})
        self.assertFalse(unknown["ok"])
        self.assertEqual(unknown["type"], "unknown")
        self.assertIn("teleport", unknown["error"]["message"])

        # Contract still usable afterwards.
        again = self.client.predict_action(_libero_style_request())
        self.assertTrue(again["ok"])

    def test_policy_error_is_reported_as_an_envelope_not_a_dropped_connection(self):
        self.policy.raise_on_next = True
        failed = self.client.predict_action(_libero_style_request())
        self.assertFalse(failed["ok"])
        self.assertEqual(failed["status"], "error")
        self.assertEqual(failed["type"], "inference_result")
        self.assertIn("stub inference failure", failed["error"]["message"])

        # An inference error must not kill the episode loop.
        recovered = self.client.predict_action(_libero_style_request())
        self.assertTrue(recovered["ok"])


class _StubFramework:
    """Minimal `baseframework` stand-in: returns normalized actions only."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def to(self, *_args, **_kwargs):
        return self

    def eval(self):
        return self

    def predict_action(self, examples, **kwargs):
        self.calls.append({"examples": examples, "kwargs": kwargs})
        return {"normalized_actions": np.zeros((BATCH, CHUNK, ACTION_DIM), dtype=np.float32)}


class _StubNormProcessor:
    """Un-normalization stand-in with a signature-visible effect."""

    unnorm_key = "libero_all"
    available_unnorm_keys = ["libero_all"]
    action_keys = ["action.delta_eef"]
    state_keys = ["state.eef"]

    def __init__(self, *_args, **kwargs) -> None:
        self.requested_key = kwargs.get("unnorm_key")

    def unapply_actions(self, normalized: np.ndarray) -> np.ndarray:
        # Deliberately not identity: proves the route is taken per batch element.
        return normalized + 3.0


def _model_cfg(action_model: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "framework": {"name": "VLA_JEPA", "action_model": action_model},
        "datasets": {"vla_data": {"data_mix": "libero_all", "obs_image_size": list(IMAGE_HW)}},
    }


class PolicyServerWrapperContractTest(unittest.TestCase):
    """Pins the wrapper's metadata and un-normalization route without a checkpoint."""

    def _build(self, action_model: Dict[str, Any], norm_keys=("libero_all",), **kwargs):
        from deployment.model_server import policy_wrapper as pw

        framework = _StubFramework()
        norm_stats = {key: {} for key in norm_keys}
        # The patches must outlive construction: `metadata` builds a norm
        # processor lazily (the eager one is cached under a different key), so a
        # construction-scoped patch would let the real loader run on attribute
        # access.
        patchers = [
            mock.patch.object(pw.baseframework, "from_pretrained", return_value=framework),
            mock.patch.object(pw, "read_mode_config", return_value=(_model_cfg(action_model), norm_stats)),
            mock.patch.object(pw, "PolicyNormProcessor", _StubNormProcessor),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        wrapper = pw.PolicyServerWrapper(ckpt_path="stub://ckpt", device="cpu", **kwargs)
        return wrapper, framework

    def test_metadata_exposes_the_fields_clients_read(self):
        wrapper, _ = self._build({"action_horizon": CHUNK})
        meta = wrapper.metadata

        self.assertEqual(meta["action_chunk_size"], CHUNK)
        self.assertEqual(meta["env"], "starvla_policy_server")
        self.assertEqual(meta["available_unnorm_keys"], ["libero_all"])
        self.assertEqual(meta["default_unnorm_key"], "libero_all")
        self.assertEqual(meta["training_data_mix"], "libero_all")
        self.assertEqual(meta["training_obs_image_size"], list(IMAGE_HW))
        self.assertEqual(meta["action_keys"], ["action.delta_eef"])
        self.assertEqual(meta["state_keys"], ["state.eef"])
        # Camera order is the client's responsibility; the server must keep saying so.
        self.assertIn("does not infer or reorder", meta["eval_image_contract"])

    def test_action_chunk_size_derivation_is_pinned(self):
        horizon_wrapper, _ = self._build({"action_horizon": 5})
        self.assertEqual(horizon_wrapper.metadata["action_chunk_size"], 5)

        window_wrapper, _ = self._build({"future_action_window_size": 6})
        self.assertEqual(window_wrapper.metadata["action_chunk_size"], 7)

        with self.assertRaisesRegex(ValueError, "action_horizon or future_action_window_size"):
            self._build({})

    def test_predict_action_unnormalizes_and_preserves_shape(self):
        wrapper, framework = self._build({"action_horizon": CHUNK})
        out = wrapper.predict_action(
            examples=[{"lang": "x"}], unnorm_key="libero_all", use_ddim=True, num_ddim_steps=10
        )

        actions = out["actions"]
        self.assertEqual(set(out), {"actions"})
        self.assertEqual(actions.shape, (BATCH, CHUNK, ACTION_DIM))
        # Framework emitted zeros; the un-normalization route adds 3.0.
        np.testing.assert_array_equal(actions, np.full((BATCH, CHUNK, ACTION_DIM), 3.0, dtype=np.float32))
        # Sampler kwargs are forwarded to the framework, unnorm_key is not.
        self.assertEqual(framework.calls[-1]["kwargs"], {"use_ddim": True, "num_ddim_steps": 10})

    def test_ambiguous_unnorm_key_is_rejected_rather_than_guessed(self):
        wrapper, _ = self._build({"action_horizon": CHUNK}, norm_keys=("libero_all", "bridge_orig"))
        self.assertIsNone(wrapper.metadata["default_unnorm_key"])
        with self.assertRaisesRegex(ValueError, "unnorm_key not specified"):
            wrapper.predict_action(examples=[{"lang": "x"}])


if __name__ == "__main__":
    unittest.main()
