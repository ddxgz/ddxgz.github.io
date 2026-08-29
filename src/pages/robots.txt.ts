import type { APIRoute } from "astro";
import config from "@/config";

const getRobotsTxt = () => `
User-agent: *
Allow: /

Host: ${config.site.url}
`;

export const GET: APIRoute = () => new Response(getRobotsTxt());
