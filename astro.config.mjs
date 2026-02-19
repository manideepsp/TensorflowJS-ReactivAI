// @ts-check
import { defineConfig } from 'astro/config';

import react from '@astrojs/react';

// https://astro.build/config
export default defineConfig({
  site: 'https://manideepsp.github.io/TensorflowJS-ReactivAI',
  base: '/TensorflowJS-ReactivAI/',
  integrations: [react()],
  output: 'static',
  vite: {
    optimizeDeps: {
      exclude: ['node-fetch', 'whatwg-url']
    },
    resolve: {
      alias: {
        // Provide browser-safe shims to avoid pulling Node polyfills
        'whatwg-url': '/src/shims/whatwg-url.ts',
        'node-fetch': '/src/shims/node-fetch.ts'
      }
    }
  }
});