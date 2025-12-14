# Contributing to Dynamical-SIL

Thank you for your interest in contributing to Dynamical-SIL!

## Development Setup

### Prerequisites

- Docker & Docker Compose
- Python 3.10+
- ROS 2 Humble or Jazzy (for native development)
- Git

### Initial Setup

```bash
# Clone repository
git clone https://github.com/Danielfoojunwei/Multi-actor.git
cd Multi-actor

# Install Python dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Start development environment
make dev-up
```

### ROS 2 Development

```bash
cd ros2_ws
colcon build --symlink-install
source install/setup.bash
```

## Code Quality

### Python

We use:
- **ruff** for linting
- **black** for formatting
- **mypy** for type checking
- **pytest** for testing

Run before committing:

```bash
make lint
make test
```

### C++

We use:
- **clang-format** for formatting
- **clang-tidy** for linting
- **ASAN/UBSAN** for sanitization

Format C++ code:

```bash
cd ros2_ws
find src -name "*.cpp" -o -name "*.hpp" | xargs clang-format -i
```

## Testing

### Unit Tests

```bash
pytest tests/unit -v
```

### Integration Tests

```bash
pytest tests/integration -v
```

### Hardware-in-the-Loop Tests

```bash
pytest tests/hil -v -m hil
```

## Pull Request Process

1. Create a feature branch: `git checkout -b feature/your-feature`
2. Make your changes
3. Add tests for new functionality
4. Ensure all tests pass: `make test`
5. Ensure linting passes: `make lint`
6. Commit with clear messages
7. Push and create a pull request

## Commit Message Guidelines

Follow conventional commits:

```
feat: add support for new camera model
fix: resolve race condition in swarm coordinator
docs: update architecture documentation
test: add integration test for federated unlearning
```

## Documentation

Update documentation when:
- Adding new features
- Changing APIs
- Modifying architecture
- Adding configuration options

Documentation lives in `docs/` and should be written in Markdown.

## Security

Report security vulnerabilities privately via GitHub Security Advisories or email the maintainers directly. Do not open public issues for security concerns.

## Code of Conduct

Be respectful, inclusive, and professional. We follow the [Contributor Covenant](https://www.contributor-covenant.org/).

## Questions?

Open a discussion on GitHub or reach out to the maintainers.
