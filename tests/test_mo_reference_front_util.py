import json
import tempfile
import unittest
from pathlib import Path

from src.utils.MO_ReferenceFrontUtil import OBJECTIVE_DEFINITION_VERSION
from src.utils.MO_ReferenceFrontUtil import compute_archive_igd
from src.utils.MO_ReferenceFrontUtil import ensure_instance_reference_front


class TestMOReferenceFrontUtil(unittest.TestCase):
    def _write_archive(self, path, instance, algorithm, items):
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "instance": instance,
            "algorithm": algorithm,
            "objectiveDefinitionVersion": OBJECTIVE_DEFINITION_VERSION,
            "items": items,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def test_build_reference_front_and_compute_igd(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_root = Path(tmp_dir) / "files" / "expresults"
            archive_dir = result_root / "pareto_archives"
            self._write_archive(
                archive_dir / "AB20-ar3-MO-a.json",
                "AB20-ar3",
                "MO_A",
                [
                    {
                        "index": 1,
                        "isFeasible": True,
                        "moObjectivesMin": [0.0, 1.0, 1.0, 1.0],
                    },
                    {
                        "index": 2,
                        "isFeasible": True,
                        "moObjectivesMin": [1.0, 1.0, 1.0, 1.0],
                    },
                ],
            )
            self._write_archive(
                archive_dir / "AB20-ar3-MO-b.json",
                "AB20-ar3",
                "MO_B",
                [
                    {
                        "index": 1,
                        "isFeasible": True,
                        "moObjectivesMin": [1.0, 0.0, 0.0, 0.0],
                    }
                ],
            )

            reference_payload = ensure_instance_reference_front("AB20-ar3", result_root=result_root, force_rebuild=True)
            self.assertEqual(reference_payload["objectiveDefinitionVersion"], OBJECTIVE_DEFINITION_VERSION)
            self.assertEqual(reference_payload["referenceFrontSize"], 2)
            self.assertEqual(reference_payload["sourceArchiveCount"], 2)

            identical_archive = {
                "items": [
                    {"isFeasible": True, "moObjectivesMin": [0.0, 1.0, 1.0, 1.0]},
                    {"isFeasible": True, "moObjectivesMin": [1.0, 0.0, 0.0, 0.0]},
                ]
            }
            partial_archive = {
                "items": [
                    {"isFeasible": True, "moObjectivesMin": [0.0, 1.0, 1.0, 1.0]},
                ]
            }

            self.assertAlmostEqual(float(compute_archive_igd(identical_archive, reference_payload)), 0.0, places=12)
            self.assertGreater(float(compute_archive_igd(partial_archive, reference_payload)), 0.0)

    def test_reference_front_ignores_archives_from_old_objective_definition(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            result_root = Path(tmp_dir) / "files" / "expresults"
            archive_dir = result_root / "pareto_archives"
            self._write_archive(
                archive_dir / "Du62-MO-new.json",
                "Du62",
                "MO_NEW",
                [{"index": 1, "isFeasible": True, "moObjectivesMin": [0.0, 1.0, 1.0, 0.5]}],
            )
            old_payload = {
                "instance": "Du62",
                "algorithm": "MO_OLD",
                "items": [{"index": 1, "isFeasible": True, "moObjectivesMin": [0.0, 1.0, 1.0, 1.0]}],
            }
            old_path = archive_dir / "Du62-MO-old.json"
            old_path.write_text(json.dumps(old_payload, ensure_ascii=False, indent=2), encoding="utf-8")

            reference_payload = ensure_instance_reference_front("Du62", result_root=result_root, force_rebuild=True)

            self.assertEqual(reference_payload["referenceFrontSize"], 1)
            self.assertEqual(reference_payload["sourceArchiveCount"], 1)
            self.assertEqual(reference_payload["sourceAlgorithms"], ["MO_NEW"])
            self.assertEqual(reference_payload["items"][0]["moObjectivesMin"], [0.0, 1.0, 1.0, 0.5])


if __name__ == "__main__":
    unittest.main()
