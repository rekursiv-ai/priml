# Security Policy

## Why this file exists

priml loads model checkpoints, runs training and data pipelines, and pulls in
a large stack of ML dependencies. Some of these paths can execute untrusted code
or process untrusted data (for example, deserializing a checkpoint). Security
reports need a private path so exploit details are not published before review.

## Reporting a vulnerability

Please report suspected security vulnerabilities privately by emailing hello@rekursiv.ai.

Include:

- Affected version or commit.
- Steps to reproduce.
- Expected impact.
- Any suggested mitigation.

Please do not open public issues for vulnerabilities until we have investigated and coordinated disclosure.

## Scope

Security reports are especially useful for:

- Unsafe model or checkpoint loading (deserialization / pickle in checkpoint or
  weight loading).
- Untrusted data-pipeline inputs that can trigger code execution or resource
  exhaustion.
- Dependency or packaging issues that affect installed users.
- Supply-chain concerns in the published wheel or its dependency set.
- SSRF or unsafe URL handling in model/dataset download paths.
