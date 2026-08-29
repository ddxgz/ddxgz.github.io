import { defineAstroPaperConfig } from "./src/types/config";

export default defineAstroPaperConfig({
  site: {
    url: "https://ddxgz.github.io/",
    title: "Cong's Notes",
    description:
      "A reboot for writing my notes and thoughts online after several years of pause.",
    author: "Cong Peng",
    profile: "https://www.linkedin.com/in/cong-peng-pcx/",
    ogImage: "og.png",
    lang: "en",
    timezone: "Europe/Stockholm",
    dir: "ltr",
  },
  posts: {
    perPage: 10,
    perIndex: 4,
    scheduledPostMargin: 15 * 60 * 1000,
  },
  features: {
    lightAndDarkMode: true,
    dynamicOgImage: true,
    showArchives: true,
    showBackButton: true,
    editPost: { enabled: false },
    search: "pagefind",
  },
  socials: [
    { name: "github", url: "https://github.com/ddxgz" },
    {
      name: "linkedin",
      url: "https://www.linkedin.com/in/cong-peng-pcx/",
    },
    { name: "mail", url: "mailto:cong.peng@actorise.com" },
  ],
  shareLinks: [
    { name: "x", url: "https://x.com/intent/post?url=" },
    { name: "telegram", url: "https://t.me/share/url?url=" },
    { name: "mail", url: "mailto:?subject=See%20this%20post&body=" },
  ],
});
