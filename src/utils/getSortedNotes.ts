import type { CollectionEntry } from "astro:content";
import noteFilter from "./noteFilter";

const getSortedNotes = (notes: CollectionEntry<"notes">[]) => {
  return notes
    .filter(noteFilter)
    .sort(
      (a, b) =>
        Math.floor(
          new Date(b.data.modDatetime ?? b.data.pubDatetime).getTime() / 1000
        ) -
        Math.floor(
          new Date(a.data.modDatetime ?? a.data.pubDatetime).getTime() / 1000
        )
    );
};

export default getSortedNotes;
