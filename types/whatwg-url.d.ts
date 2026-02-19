declare module 'whatwg-url' {
  export const URL: typeof globalThis.URL | undefined;
  export const URLSearchParams: typeof globalThis.URLSearchParams | undefined;
  export function parseURL(input: string): URL | undefined;
  const _default: {
    URL: typeof globalThis.URL | undefined;
    URLSearchParams: typeof globalThis.URLSearchParams | undefined;
    parseURL: (input: string) => URL | undefined;
  };
  export default _default;
}
