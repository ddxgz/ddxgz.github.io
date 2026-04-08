import type { CollectionEntry } from "astro:content";
import { SITE } from "@/config";

const noteFilter = ({ data }: CollectionEntry<"notes">) => {
  // This is evaluated at static build time. If a note is scheduled in the
  // future, GitHub Pages will not publish it until a later build runs.
  const isPublishTimePassed =
    Date.now() >
    new Date(data.pubDatetime).getTime() - SITE.scheduledPostMargin;
  return !data.draft && (import.meta.env.DEV || isPublishTimePassed);
};

export default noteFilter;
