declare module 'node-fetch' {
  const fetchExport: typeof globalThis.fetch;
  export default fetchExport;
  export { fetchExport as fetch };
  export const Response: typeof globalThis.Response | undefined;
  export const Request: typeof globalThis.Request | undefined;
  export const Headers: typeof globalThis.Headers | undefined;
}
