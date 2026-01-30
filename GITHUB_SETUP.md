# GitHub Setup Guide

This guide will help you upload your Photo to Video AI project to GitHub.

## Prerequisites

1. **Git installed**: Run `git --version` to verify
2. **GitHub account**: https://github.com/signup
3. **GitHub CLI (optional)**: https://cli.github.com/

## Option 1: Using GitHub CLI (Recommended)

### Step 1: Install GitHub CLI
```bash
winget install GitHub.cli
# or
choco install gh
```

### Step 2: Login to GitHub
```bash
gh auth login
```
Follow the prompts to authenticate.

### Step 3: Initialize and Push
```bash
cd photo-to-video-app

# Initialize git repository
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Photo to Video AI application

- Complete full-stack application
- FastAPI backend with video generation
- Next.js frontend with real-time updates
- Docker services (PostgreSQL, Redis, MinIO)
- Cloud and local AI support
- Comprehensive documentation"

# Create GitHub repository and push
gh repo create photo-to-video-ai --public --source=. --push
```

## Option 2: Using GitHub Website

### Step 1: Create Repository on GitHub

1. Go to https://github.com/new
2. Fill in repository details:
   - **Name**: `photo-to-video-ai`
   - **Description**: "Transform photos into AI-animated videos with natural movement"
   - **Visibility**: Public or Private
   - **DO NOT** initialize with README, .gitignore, or license (we already have these)

### Step 2: Initialize Local Repository

```bash
cd photo-to-video-app

# Initialize git
git init

# Add all files
git add .

# Create initial commit
git commit -m "Initial commit: Photo to Video AI application"
```

### Step 3: Connect and Push

Replace `YOUR_USERNAME` with your GitHub username:

```bash
# Add remote origin
git remote add origin https://github.com/YOUR_USERNAME/photo-to-video-ai.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

## Verify Upload

After pushing, verify on GitHub:
1. Go to your repository URL
2. Check all files are present
3. Verify README.md displays correctly
4. Check .gitignore is working (node_modules, venv should not be uploaded)

## What Gets Uploaded

✅ **Included**:
- Source code (frontend, backend)
- Configuration files
- Documentation (README, QUICKSTART)
- Docker configuration
- Startup scripts
- .gitignore

❌ **Excluded** (by .gitignore):
- node_modules/
- venv/
- __pycache__/
- .env files
- Build artifacts
- Temporary files
- Database files
- Model files

## Adding Repository Topics

On GitHub repository page:
1. Click ⚙️ (gear icon) next to "About"
2. Add topics:
   - `ai`
   - `video-generation`
   - `fastapi`
   - `nextjs`
   - `python`
   - `typescript`
   - `computer-vision`
   - `video-processing`
   - `stable-diffusion`

## Setting Up GitHub Actions (Optional)

Create `.github/workflows/test.yml` for CI/CD:

```yaml
name: Tests

on: [push, pull_request]

jobs:
  backend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - run: |
          cd backend
          pip install -r requirements.txt
          pytest tests/

  frontend-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-node@v3
        with:
          node-version: '18'
      - run: |
          cd frontend
          npm install
          npm test
```

## Protecting Sensitive Data

Before pushing, ensure:

1. **API keys removed** from code
   ```bash
   # Check for accidentally committed secrets
   git log --all --full-history -- backend/.env
   ```

2. **Environment variables** use .env (not committed)

3. **Passwords changed** from defaults in production

## Future Updates

After making changes:

```bash
# Stage changes
git add .

# Commit with descriptive message
git commit -m "Add: New feature description"

# Push to GitHub
git push
```

## Collaboration

To allow others to contribute:

1. **Settings** → **Manage access** → **Invite a collaborator**
2. Or accept Pull Requests from forks

## License

Add a LICENSE file:
```bash
# MIT License (recommended)
curl -o LICENSE https://raw.githubusercontent.com/licenses/license-templates/master/templates/mit.txt
```

Edit LICENSE file with your name and year.

## Repository Settings

Recommended settings on GitHub:

1. **General**:
   - ✅ Enable Issues
   - ✅ Enable Discussions
   - ✅ Enable Wiki

2. **Branches**:
   - Set `main` as default branch
   - Protect `main` branch (optional)

3. **Pages** (optional):
   - Deploy documentation
   - Source: GitHub Actions

## Common Issues

### Issue: "git: command not found"
**Solution**: Install Git from https://git-scm.com/

### Issue: "Permission denied (publickey)"
**Solution**:
```bash
# Use HTTPS instead of SSH
git remote set-url origin https://github.com/USERNAME/REPO.git
```

### Issue: "Large files"
**Solution**:
```bash
# If accidentally committed large files
git filter-repo --path-glob '*.mp4' --invert-paths
```

### Issue: "Conflicts on push"
**Solution**:
```bash
# Pull and merge first
git pull origin main --rebase
git push
```

## Next Steps

After uploading to GitHub:

1. **Add README badges**: Build status, coverage, etc.
2. **Write CONTRIBUTING.md**: Guidelines for contributors
3. **Create releases**: Tag versions (v1.0.0)
4. **Add demo**: Record GIF/video demo for README
5. **Documentation**: Expand wiki or docs/

## Resources

- [GitHub Docs](https://docs.github.com/)
- [Git Cheat Sheet](https://education.github.com/git-cheat-sheet-education.pdf)
- [Markdown Guide](https://guides.github.com/features/mastering-markdown/)

---

**Your project is now on GitHub! 🎉**

Repository will be at: `https://github.com/YOUR_USERNAME/photo-to-video-ai`
