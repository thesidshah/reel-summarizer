# Contributing to Reel Summarizer

Thank you for your interest in contributing to Reel Summarizer! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Bugs

If you find a bug, please create an issue on GitHub with:
- A clear, descriptive title
- Steps to reproduce the issue
- Expected behavior vs actual behavior
- Your environment (OS, Python version, Node version)
- Any relevant error messages or logs

### Suggesting Enhancements

Enhancement suggestions are welcome! Please create an issue with:
- A clear description of the enhancement
- Why this enhancement would be useful
- Any implementation ideas you may have

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Test your changes thoroughly
5. Commit your changes (`git commit -m 'Add some feature'`)
6. Push to your branch (`git push origin feature/your-feature-name`)
7. Open a Pull Request

### Development Setup

See the main README.md for setup instructions.

### Code Style

**Python:**
- Follow PEP 8 guidelines
- Use meaningful variable and function names
- Add docstrings to functions
- Keep functions focused and small

**TypeScript/React:**
- Use TypeScript for type safety
- Follow React best practices
- Use functional components with hooks
- Keep components focused and reusable

### Testing

Before submitting a PR:
- Test the backend endpoints manually
- Test the frontend UI flows
- Ensure no console errors
- Verify the PDF generation works correctly

### Commit Messages

- Use clear, descriptive commit messages
- Start with a verb (Add, Fix, Update, Remove, etc.)
- Keep the first line under 72 characters
- Add details in the body if needed

Example:
```
Add support for multiple languages in transcription

- Added language parameter to transcription endpoint
- Updated UI to allow language selection
- Added tests for language detection
```

## Areas for Contribution

Here are some areas where contributions would be particularly valuable:

### Features
- Support for other video platforms (TikTok, YouTube Shorts, etc.)
- Multiple language support for transcription
- Custom PDF templates
- Batch processing of multiple reels
- User authentication and saved summaries
- Database integration for storing results
- Video preview before processing
- Custom AI prompts for different content types

### Improvements
- Better error handling and user feedback
- Improved UI/UX design
- Performance optimizations
- Unit and integration tests
- Docker containerization
- CI/CD pipeline
- Better mobile responsiveness

### Documentation
- API documentation
- Code comments
- Tutorial videos
- Example use cases
- Troubleshooting guides

## Questions?

If you have questions about contributing, feel free to:
- Open an issue with your question
- Start a discussion on GitHub Discussions

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to maintain a welcoming and inclusive community.

## License

By contributing to Reel Summarizer, you agree that your contributions will be licensed under the MIT License.
