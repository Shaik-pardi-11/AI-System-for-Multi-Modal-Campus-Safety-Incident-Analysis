# Contributing

Thank you for your interest in contributing to this project. Contributions, bug reports, documentation improvements, and feature suggestions are welcome.

## Before You Start

- Check the existing issues and pull requests to avoid duplicate work.
- For significant changes, open an issue first to discuss the proposed approach.
- Keep contributions focused and explain the problem they solve.

## Reporting Bugs

When opening an issue, include:

- A clear description of the problem.
- Steps to reproduce it.
- Expected and actual behavior.
- Relevant error messages or logs.
- Your operating system, Python version, and relevant library versions.
- Hardware and inference device details when reporting GPU or OpenVINO issues.

Do not include passwords, API keys, private camera URLs, personal information, or other secrets in issue reports.

## Suggesting Features

Describe the use case, the expected behavior, and why the change would be useful. Include screenshots or sample output when they help explain the proposal.

## Making Code Changes

1. Fork the repository.
2. Create a feature branch from the latest `main` branch.
3. Make your changes using clear, descriptive names.
4. Keep the change focused and follow the existing project structure and coding style.
5. Update documentation or dependency information when your change requires it.
6. Run the relevant checks and test the affected functionality.
7. Commit your changes with a clear commit message.
8. Push your branch and open a pull request against `main`.

Example:

```bash
git checkout main
git pull origin main
git checkout -b feature/describe-your-change
```

## Testing

Before submitting a pull request:

- Run the relevant tests or scripts for the area you changed.
- Check that the application starts when your changes affect the application.
- For model or detection changes, describe the dataset split, evaluation method, and metrics used.
- Do not claim performance improvements without reproducible measurements.

Hardware-specific functionality may behave differently across CPU, NVIDIA GPU, Intel XPU, and OpenVINO devices. Clearly state which hardware and runtime you used.

## Dataset and Model Files

Do not commit large datasets, raw surveillance videos, generated outputs, or model weight files unless the maintainers explicitly request them.

Do not upload private, sensitive, or personally identifiable footage. Confirm that you have permission to use and share any sample data.

## Pull Request Guidelines

A useful pull request should:

- Explain what changed and why.
- Reference the related issue, if one exists.
- List the tests you ran and their results.
- Include screenshots for relevant user-interface changes.
- Mention known limitations or follow-up work.

Keep pull requests focused and be prepared to respond to review feedback.

## License

By submitting a contribution, you agree that your contribution is provided under the [MIT License](LICENSE), consistent with this repository's licensing terms.
