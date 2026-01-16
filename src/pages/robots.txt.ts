import type { APIRoute } from "astro";
import { SITE } from "@/config";

const getRobotsTxt = () => `
User-agent: *
Allow: /

Host: ${SITE.website}
`;

export const GET: APIRoute = () => new Response(getRobotsTxt());
