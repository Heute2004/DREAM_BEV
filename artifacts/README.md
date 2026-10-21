# Generated Artifacts

This directory is reserved for immutable model hashes, policy traces, evaluation JSON,
and publication figures produced by the commands in this project.

Only artifacts produced under the current guarded contract belong here. A result becomes canonical only after:

1. the held policy trace passes `scripts/validate_policy_contract.py`;
2. the protected compositor replay passes the causal consistency contract;
3. the evaluator, bootstrap report, and artifact hashes are generated from that protected output.
