# Contributing to Trading-ML-model

Thank you for considering a contribution! To streamline review and integration, follow these steps:

## 1. Glossary
- **PR**: Pull Request  
- **CI**: Continuous Integration

## 2. Getting Started
1. **Fork** the repo and **clone** your fork:  
   ```bash
   git clone git@github.com:fpapalardo/Trading-ML-model.git
   ```
2. **Create** a feature branch:  
   ```bash
   git checkout -b feature/short-description
   ```
3. **Install** dependencies (virtualenv/conda):  
   ```bash
   pip install -r requirements.txt
   ```

## 3. Coding Standards
- Follow **PEP8**.  
- Docstrings in **Google** style.  
- Use **type hints** for all public functions.

## 4. Testing
- Place tests in `tests/`.  
- Before submitting:  
  ```bash
  pytest --maxfail=1 --disable-warnings -q
  ```

## 5. Commit Messages
- Imperative tense: “Add feature” _not_ “Added feature.”  
- Reference issues: `Fixes #123`.

## 6. Branch & PR Workflow
1. Push your branch to your fork.  
2. Open a PR against `main`.  
3. Ensure **CI** passes.  
4. Request reviews; address comments promptly.
