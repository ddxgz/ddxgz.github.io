export const SITE = {
  website: "https://ddxgz.github.io", 
  author: "Cong Peng",
  profile: "https://www.linkedin.com/in/cong-peng-pcx/",
  desc: "A reboot for writing my notes and thoughts online after several years of pause.",
  title: "Cong's Notes",
  ogImage: "",
  lightAndDarkMode: true,
  postPerIndex: 4,
  postPerPage: 4,
  scheduledPostMargin: 15 * 60 * 1000, // 15 minutes
  showArchives: true,
  showBackButton: true, // show back button in post detail
  editPost: {
    enabled: false,
    text: "Edit page",
    url: "https://github.com/ddxgz/pcx-astro-paper-github/edit/main/",
  },
  dynamicOgImage: true,
  dir: "ltr", // "rtl" | "auto"
  lang: "en", // html lang code. Set this empty and default will be "en"
  timezone: "Europe/Stockholm", // Default global timezone (IANA format) https://en.wikipedia.org/wiki/List_of_tz_database_time_zones
} as const;
