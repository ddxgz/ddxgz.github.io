import kebabcase from "lodash.kebabcase";

export const slugifyStr = (str: string): string => kebabcase(str);

export const slugifyAll = (arr: string[]) => arr.map(str => slugifyStr(str));
