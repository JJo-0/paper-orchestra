import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from autoraters.agent_review import perform_review_agentreview
from paper_writing_cli import main
from methods.agents.content_refinement_agent import ContentRefinementAgent
from methods.agents.literature_review_agent import HybridLiteratureAgent

MODELS = ("codex/gpt-6-astra", "claude/sonnet", "gemini-3.8-flash")


class ManuscriptWorkflow(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.raw = self.root / "materials"
        self.raw.mkdir()
        self.args = ["paper_writing_cli.py", "--raw_materials_dir", str(self.raw),
                     "--latex_template_dir", "templates/iclr2025",
                     "--output_dir", str(self.root / "draft")]

    def test_single_model_selects_full_pipeline_in_both_modes(self):
        for model in MODELS:
            for plotting in (True, False):
                module = "methods.paper_writer_with_plotting" if plotting else "methods.paper_writer"
                with self.subTest(model=model, plotting=plotting), patch(
                    module + ".write_single_paper", return_value=True
                ) as write, patch.object(sys, "argv", self.args + [
                    "--model", model, "--use_plotting", str(plotting)
                ]):
                    main()
                    options = write.call_args.kwargs
                    self.assertEqual(options["writer_model_name"], model)
                    self.assertEqual(options["reflection_model_name"], model)
                    if plotting:
                        self.assertEqual(options["plotting_model_name"], model)

    def test_mixed_runs_require_explicit_roles_and_failure_reaches_shell(self):
        with patch.object(sys, "argv", self.args + [
            "--model", MODELS[0], "--reflection_model_name", MODELS[1]
        ]), self.assertRaises(SystemExit) as ambiguous:
            main()
        self.assertEqual(ambiguous.exception.code, 2)
        with patch("methods.paper_writer.write_single_paper", return_value=False) as write, patch.object(
            sys, "argv", self.args + ["--writer_model_name", MODELS[0], "--reflection_model_name", MODELS[1]]
        ), self.assertRaises(SystemExit) as failed:
            main()
        self.assertEqual(failed.exception.code, 1)
        self.assertEqual(write.call_args.kwargs["reflection_model_name"], MODELS[1])

    def test_screened_sources_are_shared_and_synthesized_by_each_writer(self):
        entry = dict(key="study", title="Example Study", venue="Example", authors=["A. Lee"],
                     year=2024, summary="A screened finding.")
        screened = self.raw / "screened_literature.json"
        screened.write_text(json.dumps([entry]))
        outlines = self.root / "outline.json"
        outlines.write_text("{}")
        for model in MODELS:
            agent = HybridLiteratureAgent(str(self.raw / "idea.md"), "", "", "", self.root, model)
            with patch.object(agent, "_execute_tasks_in_parallel") as search, patch(
                "methods.agents.literature_review_agent.call_gemini_with_contents",
                return_value={"parsed_response": "Related work"}
            ) as synthesis:
                agent.run(outlines, "2026-11")
                search.assert_not_called()
                self.assertEqual(synthesis.call_args.kwargs["model_name"], model)
                library = json.loads((self.root / "citation_map.json").read_text())
                self.assertEqual(library["study"]["abstract"], entry["summary"])
        for invalid in ([], [entry, entry], [dict(entry, year=2027)]):
            screened.write_text(json.dumps(invalid))
            with self.assertRaises(ValueError):
                agent.run(outlines, "2026-11")

    def test_discovery_honors_exact_gemini_model(self):
        agent = HybridLiteratureAgent("", "", "", "", self.root, MODELS[2])
        with patch("methods.agents.literature_review_agent.call_gemini_with_contents",
                   return_value={"parsed_response": {"candidates": []}}) as search:
            agent._discover_candidates({"focus": "example", "context": "related work"}, {}, "2026-11")
        self.assertEqual(search.call_args.kwargs["model_name"], MODELS[2])

    def test_every_peer_review_and_meta_review_use_selected_model(self):
        for model in MODELS:
            with self.subTest(model=model), patch(
                "autoraters.agent_review.call_gemini_with_text_prompt",
                return_value={"parsed_response": {"Overall": 7}}
            ) as request:
                self.assertEqual(perform_review_agentreview("draft", model)["Overall"], 7)
                self.assertEqual([call.kwargs["model_name"] for call in request.call_args_list], [model] * 4)

    def test_no_successful_baseline_review_on_provider_failure(self):
        agent = ContentRefinementAgent("", "", "", model_name=MODELS[1], work_dir=self.root)
        with patch("methods.agents.content_refinement_agent.load_paper", return_value="draft"), patch(
            "methods.agents.content_refinement_agent.perform_review_agentreview", side_effect=RuntimeError("quota")
        ), self.assertRaisesRegex(RuntimeError, "quota"):
            agent._get_peer_review("draft.pdf", "initial")
        self.assertFalse((self.root / "peer_reviews/review_v0.json").exists())
