import type { APIRoute } from "astro";
import { SITE } from "@/config";

const baseUrl = new URL(SITE.website);
const toUrl = (path: string) => new URL(path, baseUrl).href;

const sections = [
  { label: "Home", path: "/" },
  { label: "About", path: "/about/" },
  ...(SITE.showArchives ? [{ label: "Archives", path: "/archives/" }] : []),
  { label: "Tags", path: "/tags/" },
  { label: "Publications", path: "/publications/" },
  { label: "Search", path: "/search/" },
];

const resources = [
  { label: "RSS", path: "/rss.xml" },
  { label: "LLMs.txt", path: "/llms.txt" },
  { label: "AI.txt", path: "/ai.txt" },
];

const topics = SITE.llms.topics ?? [];

const getLlmsTxt = () => {
  const sectionLines = sections
    .map(section => `- [${section.label}](${toUrl(section.path)})`)
    .join("\n");

  const resourceLines = resources
    .map(resource => `- [${resource.label}](${toUrl(resource.path)})`)
    .join("\n");

  const topicLines =
    topics.length > 0 ? `\n## Topics\n- ${topics.join("\n- ")}\n` : "";

  return `# ${SITE.title}
> ${SITE.desc}

## About
- Website: ${SITE.website}
- Language: ${SITE.lang ?? "en"}
- Author: ${SITE.author}
- Profile: ${SITE.profile}

## About Me
- Engineer / Architect / Builder based in Stockholm
- Builds agentic AI applications and cloud data & AI platforms
- Leads architecture and hands-on delivery from MVP to production
- Currently building Actorise: https://www.actorise.com

## Content
- Articles and notes in the blog
- Publications list

## Sections
${sectionLines}
${topicLines}

## Usage
- You may crawl and summarize public pages
- Attribute content to ${SITE.title} and link to canonical URLs

## Resources
${resourceLines}
`;
};

export const GET: APIRoute = () =>
  new Response(getLlmsTxt(), {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
    },
  });
