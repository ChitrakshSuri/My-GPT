# Deploying

The live site runs on **Render's free plan** from the `Dockerfile` and `render.yaml` in this repo.

## Why there is a `deploy` branch

The trained weights `ai/gpt.pt` (52 MB) are git-ignored, so they are not on `main`. The `deploy` branch is `main`
plus that file, and Render builds from it.

To update the live site after changing `main`, work in a second folder for `deploy`:

```bash
git worktree add ../My-GPT-deploy deploy     # once
cd ../My-GPT-deploy
git merge main
git push origin deploy
```

Do not run `git checkout deploy` in your normal folder. `ai/gpt.pt` is tracked on `deploy` but not on `main`, so switching
branches there deletes your local copy of the weights.

Render redeploys by itself when the `deploy` branch changes.

## Setting it up from scratch

1. Sign in at https://render.com with GitHub and allow access to the repo.
2. Choose **New, Blueprint**, pick this repo, give the blueprint a name, and click **Deploy Blueprint**.
3. Render reads `render.yaml`, builds the `Dockerfile` (the first build takes a few minutes), and gives you a public
   `onrender.com` URL.

## Free plan limits

- The container gets only a small share of a CPU, so a trace takes several seconds and a long chat reply takes a while.
- It goes to sleep after 15 minutes without visitors. Waking it takes up to a minute.

## Run the same container locally

```bash
docker compose up --build        # http://localhost:7860
```
