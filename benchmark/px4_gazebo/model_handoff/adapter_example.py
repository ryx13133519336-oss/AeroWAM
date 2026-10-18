"""Transport-only hover example; this is not a benchmark baseline/model."""

from .protocol import ModelObservation, hover_action


def create_policy():
    def policy(payload):
        observation = ModelObservation.from_wire(payload)
        # External VLN/VLA models receive observation.instruction unchanged.
        # Replace this hover with model inference and return body-FLU deltas.
        assert observation.instruction
        return hover_action().to_wire()
    return policy


def create_grounded_policy():
    """Transport-only opt-in example; emits no fabricated visual evidence."""
    from .grounded_interface import GroundedAction, GroundedAdapter
    return GroundedAdapter(lambda observation: GroundedAction(hover_action()))
