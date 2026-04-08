# Cong's Notes

Personal knowledge garden built with [Astro](https://astro.build/) and [AstroPaper](https://github.com/satnaing/astro-paper). The goal of this repo is to publish essays, research notes, and project summaries.

## Local development

```bash
pnpm install
pnpm run dev # http://localhost:4321
```

Useful scripts:

- `pnpm run build` – generate the production site in `dist/`
- `pnpm run preview` – preview the most recent build locally
- `pnpm run lint` – run ESLint across the project
- `pnpm run format` – format source files with Prettier

## Content structure

- Site settings live in `src/config.ts` and `src/constants.ts`
- Blog posts live in `src/data/blog/`
- UI components/layouts live under `src/components` and `src/layouts`
- Static assets belong in `public/`

## Deployment

The site is designed for static hosting (GitHub Pages, Cloudflare Pages, etc.). Run `pnpm run build` and deploy the contents of `dist/` to your host of choice. Continuous deployment from the `master` branch keeps https://ddxgz.github.io up to date.

### Scheduled publishing note

- Notes and posts are filtered at build time based on `pubDatetime` and `scheduledPostMargin`.
- GitHub Pages only republishes on `push` or manual workflow runs, so a note can still be missing from the live site if its publish time has not passed when the last build ran.
- Use the correct timezone offset in `pubDatetime`. For Stockholm local time that usually means `+01:00` in winter and `+02:00` in summer.
- If a scheduled note is missing after its publish time, rerun the Pages workflow or push a new commit to trigger a fresh build.

## Format code before commit

```bash
pnpm run format:check --write
```
