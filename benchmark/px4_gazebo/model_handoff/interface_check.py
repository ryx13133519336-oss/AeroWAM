"""Validate saved external-adapter requests/responses without starting flight."""

import argparse
import io
import json
from pathlib import Path

from .decision_interface import (DECISION_PROTOCOL_VERSION, CandidatePrediction, DecisionRequest, validate_predictions)
from .protocol import ModelJointAction, ModelObservation


def check(request: dict, response: dict | None = None) -> str:
    decision = request.get("protocol") == DECISION_PROTOCOL_VERSION
    parsed = DecisionRequest.from_wire(request) if decision else ModelObservation.from_wire(request)
    observations = parsed.history if decision else (parsed,)
    from PIL import Image
    for observation in observations:
        for payload in observation.rgb_jpeg.values():
            with Image.open(io.BytesIO(payload)) as image:
                if image.format != "JPEG" or image.size != (256, 256) or image.mode != "RGB":
                    raise ValueError("Expected 256x256 RGB JPEG observation")
                image.load()
    if response is not None:
        if decision:
            if (set(response) != {"protocol", "decision_id", "predictions"}
                    or response["protocol"] != DECISION_PROTOCOL_VERSION
                    or response["decision_id"] != parsed.decision_id):
                raise ValueError("Mismatched decision response")
            validate_predictions(parsed, tuple(CandidatePrediction.from_wire(row) for row in response["predictions"]))
        else:
            ModelJointAction.from_wire(response)
    return "decision" if decision else "policy"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--response", type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text(encoding="utf-8"))
    response = json.loads(args.response.read_text(encoding="utf-8")) if args.response else None
    print(f"{check(request, response)} interface check: PASS (structure/images only, not model capability)")


if __name__ == "__main__":
    main()
