
[//]: # "SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved."
[//]: # "SPDX-License-Identifier: Apache-2.0"
[//]: # ""
[//]: # "Licensed under the Apache License, Version 2.0 (the 'License');"
[//]: # "you may not use this file except in compliance with the License."
[//]: # "You may obtain a copy of the License at"
[//]: # "http://www.apache.org/licenses/LICENSE-2.0"
[//]: # ""
[//]: # "Unless required by applicable law or agreed to in writing, software"
[//]: # "distributed under the License is distributed on an 'AS IS' BASIS"
[//]: # "WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied."
[//]: # "See the License for the specific language governing permissions and"
[//]: # "limitations under the License."

# Security

NVIDIA is dedicated to the security and trust of our software products and
services, including all source code repositories managed through our
organization.

If you need to report a security issue, please use the appropriate contact
points outlined below.
**Please do not report security vulnerabilities through GitHub/GitLab.**

## Reporting Potential Security Vulnerability in an NVIDIA Product

---
To report a potential security vulnerability in any NVIDIA product:

- Web: [Security Vulnerability Submission Form](https://www.nvidia.com/object/submit-security-vulnerability.html)
- E-Mail: psirt@nvidia.com
  - We encourage you to use the following PGP key for secure email communication: [NVIDIA public PGP Key for communication](https://www.nvidia.com/en-us/security/pgp-key)
  - Please include the following information:
  - Product/Driver name and version/branch that contains the vulnerability

## Code Static Analysis

In our commitment to maintaining the highest standards of code quality and security, we have enabled GitHub's Code Static Analysis scanning on our repositories. Static Analysis is a powerful tool for analyzing the codebase for potential vulnerabilities.

- Scope: CodeQL scanning is activated across all branches of this repository.
- Frequency: Scans are conducted regularly on new commits to ensure continuous integration and delivery are secure.
- Results Handling: Any identified vulnerabilities or code issues are reviewed and addressed promptly by our development team.
- Community Contribution: We welcome contributions to enhance our CodeQL queries. If you have suggestions or improvements, please submit a pull request or contact us via the outlined channels.

## Secrets Scanning

To further bolster our repository's security, we have implemented GitHub's secrets scanning feature. This feature helps detect and prevent accidental commits of sensitive information such as passwords, private keys, and API tokens.

- Active Scanning: Secrets scanning is active on all branches of this repository.
- Alerts and Notifications: In the event that a potential secret is committed to the repository, an alert is generated. These alerts are reviewed and addressed swiftly by our security team.
- Prevention and Education: We continuously educate our contributors about best practices in handling secrets and sensitive data. We encourage the use of environment variables and secure vaults for managing secrets.
