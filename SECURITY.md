# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, report them privately:

1. **GitHub Security Advisories**: Use the "Security" tab in this repository
2. **Email**: Contact the maintainers directly (see MAINTAINERS.md)

Please include:

- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Depends on severity
  - Critical: 7-14 days
  - High: 14-30 days
  - Medium: 30-60 days
  - Low: Best effort

## Security Best Practices

### For Developers

1. **No Secrets in Code**: Use environment variables or secret managers
2. **Dependency Scanning**: Run `make lint` to check for known vulnerabilities
3. **SBOM Generation**: Included in CI pipeline
4. **Signed Artifacts**: All CSA artifacts must be cryptographically signed
5. **Input Validation**: Validate all external inputs (camera streams, network data)

### For Operators

1. **Network Isolation**: Run components in isolated network segments
2. **Least Privilege**: Use minimal permissions for all services
3. **Secret Management**: Use Vault, AWS Secrets Manager, or similar
4. **Regular Updates**: Keep dependencies and base images updated
5. **Monitoring**: Enable all observability features

### For Users

1. **HTTPS Only**: Always use TLS for registry communication
2. **Verify Artifacts**: Check CSA signatures before deployment
3. **Privacy Modes**: Choose appropriate privacy mode for your threat model
4. **Audit Logs**: Review deployment and rollback logs regularly

## Threat Model

See [docs/THREAT_MODEL.md](docs/THREAT_MODEL.md) for detailed threat analysis.

## Privacy Considerations

This system processes sensitive demonstration data. Follow privacy guidelines in [docs/PRIVACY.md](docs/PRIVACY.md).

## Disclosure Policy

Once a vulnerability is fixed:

1. We will publish a security advisory
2. Credit will be given to the reporter (unless they prefer anonymity)
3. A CVE may be requested for significant vulnerabilities

## Contact

For urgent security matters, contact the security team directly through GitHub Security Advisories.
