# Security policy

Please report vulnerabilities privately through GitHub's
"Report a vulnerability" (Security → Advisories) rather than a public issue.

Scope notes:

* Qubolts never needs user-level data. Resolvers consume aggregated coefficients
  (segment sizes, overlap counts, costs). Do not pass personal data into problem
  objects or run logs.
* Remote backends send the QUBO (numbers and variable labels) to the provider.
  Use opaque variable labels if segment names are sensitive.
* Keep solver credentials (e.g. `DWAVE_API_TOKEN`) in a secret store such as
  Azure Key Vault, not in notebooks or repositories.
