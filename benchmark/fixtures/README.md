# MVP fixtures

`mvp_tasks.json` freezes the first six task templates and the three evaluation seeds (`0, 1, 2`). It is deliberately a task-level manifest rather than a simulator asset dump: the AirSim/TravelUAV adapter will resolve `map_id`, target instances, and occlusion layouts to concrete scene assets.

The source/target boundary is part of the fixture. The first two inspection templates are source-train tasks; handoff and search templates are target zero-shot/few-shot tasks. This is a scaffold for the transfer protocol, not a claim that six templates are statistically sufficient for a final paper.

Run the dependency-free smoke check from the repository root:

```powershell
python benchmark/tools/validate_contract.py
```
