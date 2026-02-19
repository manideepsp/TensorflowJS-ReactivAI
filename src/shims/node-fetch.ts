// Browser-safe shim for node-fetch; relies on global fetch
// Exports default fetch and named exports minimal enough for consumers

const fetchShim: typeof fetch = (...args) => fetch(...args);

export default fetchShim;
export { fetchShim as fetch };

// Placeholder Response/Request/Headers re-exports (from global)
export const Response = (typeof window !== 'undefined' ? window.Response : undefined) as
  | typeof window.Response
  | undefined;
export const Request = (typeof window !== 'undefined' ? window.Request : undefined) as
  | typeof window.Request
  | undefined;
export const Headers = (typeof window !== 'undefined' ? window.Headers : undefined) as
  | typeof window.Headers
  | undefined;
