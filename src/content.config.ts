import { defineCollection, z } from "astro:content";
import { glob } from "astro/loaders";
import { SITE } from "@/config";

export const BLOG_PATH = "src/data/blog";
export const PUBLICATIONS_PATH = "src/data/publications";

const blog = defineCollection({
  loader: glob({ pattern: "**/[^_]*.md", base: `./${BLOG_PATH}` }),
  schema: ({ image }) =>
    z.object({
      author: z.string().default(SITE.author),
      pubDatetime: z.date(),
      modDatetime: z.date().optional().nullable(),
      title: z.string(),
      featured: z.boolean().optional(),
      draft: z.boolean().optional(),
      tags: z.array(z.string()).default(["others"]),
      ogImage: image().or(z.string()).optional(),
      description: z.string(),
      canonicalURL: z.string().optional(),
      hideEditPost: z.boolean().optional(),
      timezone: z.string().optional(),
    }),
});

const publications = defineCollection({
  loader: glob({ pattern: "**/[^_]*.md", base: `./${PUBLICATIONS_PATH}` }),
  schema: z.object({
    title: z.string(),
    description: z.string().optional(),
    pubDatetime: z.date(),
    authors: z.array(z.string()).default([]),
    type: z.string().optional(),
    pdfUrl: z.string().url().optional(),
    doiUrl: z.string().url().optional(),
  }),
});

export const collections = { blog, publications };
