import json
import re
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

from stockmind_telegram_mvp.app.ai import (
    CHART_PROMPT_SHA256,
    CHART_PROMPT_VERSION,
    ResearchAI,
)


class ChartPredictionProvenanceTests(unittest.IsolatedAsyncioTestCase):
    def _research(self, response, requested_model):
        research = object.__new__(ResearchAI)
        research.vision_model = requested_model
        create = AsyncMock(return_value=response)
        research.client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        return research, create

    async def test_chart_analysis_returns_provider_model_prompt_and_timestamp(self):
        created = int(datetime(2026, 11, 5, 20, 24, tzinfo=timezone.utc).timestamp())
        response = SimpleNamespace(
            model="vision-model-2026-11-05",
            created=created,
            system_fingerprint="fp-test",
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"analysis": "Visible structure only.", "zones": []})
                    )
                )
            ],
        )
        research, create = self._research(response, "gpt-6-luna-test")

        result = await research.analyze_chart_image([], "Map DNS", b"image-bytes", "image/png")

        metadata = result["prediction_metadata"]
        self.assertEqual(metadata["provider"], "openai")
        self.assertEqual(metadata["model"], "vision-model-2026-11-05")
        self.assertEqual(metadata["requested_model"], "gpt-6-luna-test")
        self.assertEqual(metadata["prompt_version"], CHART_PROMPT_VERSION)
        self.assertEqual(metadata["prompt_sha256"], CHART_PROMPT_SHA256)
        self.assertEqual(metadata["generated_at"], "2026-11-05T20:24:00Z")
        self.assertEqual(metadata["system_fingerprint"], "fp-test")
        self.assertEqual(metadata["settings"]["temperature"], 0.1)
        self.assertEqual(metadata["settings"]["image_detail"], "high")
        self.assertEqual(metadata["settings"]["reasoning_effort"], "none")
        self.assertEqual(create.await_args.kwargs["reasoning_effort"], "none")
        self.assertEqual(
            create.await_args.kwargs["messages"][-1]["content"][1]["image_url"]["detail"],
            "high",
        )

    async def test_missing_provider_model_or_timestamp_is_not_guessed(self):
        response = SimpleNamespace(
            model=None,
            created=None,
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content=json.dumps({"analysis": "Uncertain metadata.", "zones": []})
                    )
                )
            ],
        )
        research, _ = self._research(response, "configured-alias")

        result = await research.analyze_chart_image([], "", b"image-bytes", "image/png")

        self.assertIsNone(result["prediction_metadata"]["model"])
        self.assertIsNone(result["prediction_metadata"]["generated_at"])

    def test_prompt_version_and_static_prompt_hash_are_present(self):
        self.assertTrue(CHART_PROMPT_VERSION)
        self.assertRegex(CHART_PROMPT_SHA256, re.compile(r"^[0-9a-f]{64}$"))


if __name__ == "__main__":
    unittest.main()
