import { PUBLICATIONS_PATH } from "@/content.config";
import { slugifyStr } from "./slugify";

/**
 * Get full path of a publication page
 * @param id - id of the publication (aka slug)
 * @param filePath - the publication full file location
 * @param includeBase - whether to include `/publications` in return value
 * @returns publication path
 */
export function getPublicationPath(
  id: string,
  filePath: string | undefined,
  includeBase = true
) {
  const pathSegments = filePath
    ?.replace(PUBLICATIONS_PATH, "")
    .split("/")
    .filter(path => path !== "")
    .filter(path => !path.startsWith("_"))
    .slice(0, -1)
    .map(segment => slugifyStr(segment));

  const basePath = includeBase ? "/publications" : "";

  // Making sure `id` does not contain the directory
  const pubId = id.split("/");
  const slug = pubId.length > 0 ? pubId.slice(-1) : pubId;

  if (!pathSegments || pathSegments.length < 1) {
    return [basePath, slug].join("/");
  }

  return [basePath, ...pathSegments, slug].join("/");
}

