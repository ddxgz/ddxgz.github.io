import type { APIRoute } from "astro";
import config from "@/config";

const getAiTxt = () => `# AI & LLM Access
Site: ${config.site.url}
Owner: ${config.site.author}
Profile: ${config.site.profile}

Preferred usage:
- Crawl and index publicly accessible pages
- Attribute content to ${config.site.title} and link to the canonical URL
`;

export const GET: APIRoute = () =>
  new Response(getAiTxt(), {
    headers: {
      "Content-Type": "text/plain; charset=utf-8",
    },
  });
