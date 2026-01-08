# Cong's Notes

Personal knowledge garden built with [Astro](https://astro.build/) and the AstroPaper theme. The goal of this repo is to publish essays, research notes, and project summaries without carrying over the upstream demo content.

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

The site is designed for static hosting (GitHub Pages, Cloudflare Pages, etc.). Run `pnpm run build` and deploy the contents of `dist/` to your host of choice. Continuous deployment from the `main` branch keeps https://ddxgz.github.io up to date.
