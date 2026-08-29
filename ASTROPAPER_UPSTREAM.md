# AstroPaper upstream baseline

This site is migrated to AstroPaper `v6.1.0` at commit
`4c33a60529f9c443145a89fe526ff231c009272d`.

Use the dependency set from that tag as one tested baseline. In particular:

- Astro `^6.4.2`
- Node.js `>=22.12.0` with Node.js 24 in CI
- pnpm `11.3.0`

Do not independently bump Astro to a newer major without checking the matching
AstroPaper release and migration notes.

## Local surfaces to preserve

- `astro-paper.config.ts` contains settings supported by AstroPaper.
- `src/site-extensions.config.ts` contains local settings that are intentionally
  outside AstroPaper's configuration schema.
- Posts and pages use the v6 collection locations under `src/content/`.
- Notes and publications remain custom collections under `src/content/`.
- `.org` files beside posts are retained as source only; the loader excludes
  them from publication.
- The homepage, Notes navigation, publication pages, AI/LLM text endpoints,
  SEO metadata, Pagefind build, dynamic OG images, and GitHub Pages deployment
  are local behavior.

## Future upgrade method

1. Fetch the next signed/tagged AstroPaper release.
2. Compare it to `v6.1.0`, not to the untagged upstream default branch.
3. Apply upstream core changes first.
4. Reapply and validate the local surfaces above.
5. Keep public post, note, publication, feed, and endpoint URLs stable.
