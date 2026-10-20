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

## Keeping the site awake (optional)

The workflow `.github/workflows/keep-alive.yml` calls `/api/health` every 5 minutes, which counts as traffic, so the
free instance does not spin down. Run it once by hand to check it: Actions tab, "Keep the site awake", **Run workflow**.

Know what it costs:

- **Free hours.** Render gives 750 free instance hours per month to the whole workspace, and a running service uses
  them. An always-awake service uses about 720 to 744 hours, which leaves almost no margin. If the hours run out,
  Render suspends all free services until the next month. Another free web service in the same workspace shares the 750.
- **Render's attitude.** A Render staff member wrote in
  [a community thread](https://community.render.com/t/will-using-cron-jobs-to-hit-free-tier-web-service-every-13-14-minutes-use-up-my-free-instance-hours/23630)
  that deliberately keeping a free service awake goes against the spirit of the free tier. That is a forum answer, not a
  formal rule, but keep it in mind.
- **It is not perfect.** GitHub can run scheduled jobs late or skip some, and it disables scheduled workflows in a public
  repo after 60 days without any repository activity. The site can still sleep now and then.

To pause it, open the workflow in the Actions tab and choose **Disable workflow**. To keep the site awake for only part
of the day, limit the hours in the `cron` line. For example `"2-59/5 2-18 * * *"` runs from 02:00 to 18:59 UTC, which
is about 07:30 to 00:30 in India and uses roughly 500 hours a month.

A free uptime monitor such as UptimeRobot, set to check the health URL every 5 minutes, does the same job and also
emails you when the site is down.

## Free plan limits

- The container gets only a small share of a CPU, so a trace takes several seconds and a long chat reply takes a while.
- It goes to sleep after 15 minutes without visitors. Waking it takes up to a minute.

## Run the same container locally

```bash
docker compose up --build        # http://localhost:7860
```
