import { defineThemeConfig } from 'vuepress-theme-plume'
import { enNavbar, zhNavbar } from './navbar'
import { enNotes, zhNotes } from './notes'

/**
 * @see https://theme-plume.vuejs.press/config/basic/
 */
export default defineThemeConfig({
  logo: 'https://theme-plume.vuejs.press/plume.png',
  // your git repo url
  docsRepo: '',
  docsDir: 'docs',

  appearance: true,

  social: [
    { icon: 'github', link: '/' },
    { icon: 'weibo', link: '/' },
    { icon: 'qq', link: '/' },
    { icon: 'facebook', link: '/' },
  ],

  locales: {
    '/': {
      profile: {
        avatar: 'https://theme-plume.vuejs.press/plume.png',
        name: 'Super_GXW',
        description: '一个简约的，记录阅读、学习笔记的个人博客',
        // circle: true,
        // location: '',
        // organization: '',
      },

      navbar: zhNavbar,
      notes: zhNotes,
    },
    '/en/': {
      profile: {
        avatar: 'https://theme-plume.vuejs.press/plume.png',
        name: 'Super_GXW',
        description: 'A simple personal blog that records reading and learning notes',
        // circle: true,
        // location: '',
        // organization: '',
      },

      navbar: enNavbar,
      notes: enNotes,
    },
  },
})
