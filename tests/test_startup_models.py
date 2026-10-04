"""First-run GPU tiers, prediction heads and loading admission."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from ml_stack.fleet import routes, serving, startup_models
from ml_stack.fleet.models import Downloads, Getting
from ml_stack.hub.probe import GIB, Gpu, MachineMemory


class StartupModelTests(unittest.TestCase):
    def machine(self, capacity, free, ram=64):
        return MachineMemory(total_ram=ram * GIB, available_ram=ram * GIB,
                             gpus=(Gpu("card", int(capacity * GIB), int(free * GIB)),))

    def test_nominal_card_capacity_selects_large_model_even_when_busy(self):
        result = startup_models.choices(machine=self.machine(24564 / 1024, 4))
        selected = next(row for row in result["models"] if row["recommended"])
        self.assertEqual(selected["params_b"], 27)
        self.assertIn("UD-Q4_K_XL", selected["ref"])
        self.assertIn("/MTP/mtp-Qwen3.8-27B", selected["draft_ref"])
        self.assertFalse(selected["fits_now"])

    def test_small_gpu_does_not_take_tier_from_host_ram(self):
        result = startup_models.choices(machine=self.machine(16, 16, ram=256))
        selected = next(row for row in result["models"] if row["recommended"])
        self.assertEqual(selected["name"], "Qwen3.5 9B")
        self.assertFalse(any(row["params_b"] == 27 for row in result["models"]))

    def test_two_small_cards_do_not_combine_into_large_tier(self):
        machine = MachineMemory(total_ram=64 * GIB, available_ram=64 * GIB,
                                gpus=(Gpu("a", 12 * GIB, 12 * GIB),
                                      Gpu("b", 12 * GIB, 12 * GIB)))
        result = startup_models.choices(machine=machine)
        self.assertFalse(any(row["params_b"] == 27 for row in result["models"]))

    def test_disk_offer_accounts_for_prediction_head(self):
        result = startup_models.choices(machine=self.machine(24, 24), disk_gb=17)
        self.assertFalse(any(row["params_b"] == 27 for row in result["models"]))

    def test_failed_prediction_head_is_a_failed_download(self):
        models = Mock()
        models.ensure.return_value = Mock(name="model", size=1024)
        models.ensure_draft.side_effect = OSError("head download interrupted")
        downloads = Downloads(models)
        row = Getting(id="test", name="model", source="hf:example/model/model.gguf")
        downloads._run(row, None, True, "hf:example/model/MTP/mtp-head.gguf")
        self.assertEqual(row.state, "failed")
        self.assertIn("head download interrupted", row.error)

    def test_head_is_typed_and_free_memory_refuses_before_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / "model.gguf"
            draft = Path(directory) / "model.draft.gguf"
            draft.touch()
            manager = Mock()
            with patch.object(serving, "read_gguf_header", return_value={
                    "general.architecture": "qwen3_5", "qwen3_5.nextn_predict_layers": 1}), patch.object(
                    serving, "estimate"), patch.object(serving, "verdict", return_value="red"):
                with self.assertRaises(serving.NoRoom):
                    serving.start_model(directory, model, manager=manager)
                manager.lease.assert_not_called()
            with patch.object(serving, "read_gguf_header", return_value={
                    "general.architecture": "qwen3_5", "qwen3_5.nextn_predict_layers": 1}), patch.object(
                    serving, "estimate"), patch.object(serving, "verdict", return_value="green"):
                serving.start_model(directory, model, manager=manager)
                spec = manager.lease.call_args.args[0]
                self.assertEqual(spec.spec_type, "draft-mtp")
                self.assertEqual(spec.draft, str(draft))
                self.assertEqual(spec.spec_draft_ngl, 99)
                self.assertFalse(spec.extra_args)

    def test_startup_route_refuses_untrusted_requests_and_ignores_query_paths(self):
        handler = Mock(path="/ui/models/startup?source=file:///outside&name=../../outside",
                       command="GET", client_address=("192.0.2.1", 1000))
        handler.headers = {}
        ui = Mock()
        ui.authed.return_value = False
        ui.may_setup.return_value = "remote setup refused"
        with patch.object(routes, "choices") as offered, patch.object(
                routes, "in_cluster", return_value=False), patch.object(routes.Router, "send") as send:
            routes.Router(ui, handler).run()
            self.assertEqual(send.call_args.args[0], 403)
            offered.assert_not_called()
            handler.headers = {routes.UI_HEADER: "1"}
            routes.Router(ui, handler).run()
            self.assertEqual(send.call_args.args[0], 403)
            offered.assert_not_called()
            ui.authed.return_value = True
            ui.models = None
            routes.Router(ui, handler).run()
            offered.assert_called_once_with(disk_gb=0, installed=())


if __name__ == "__main__":
    unittest.main()
