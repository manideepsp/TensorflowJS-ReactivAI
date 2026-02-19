// Lightweight browser-safe shim for whatwg-url to satisfy node-fetch/consumer imports
const URLShim = (typeof window !== 'undefined' ? window.URL : undefined) as typeof URL | undefined;
const URLSearchParamsShim = (typeof window !== 'undefined' ? window.URLSearchParams : undefined) as
	| typeof URLSearchParams
	| undefined;

// Minimal helpers matching whatwg-url surface we actually touch
const parseURL = (input: string) => (URLShim ? new URLShim(input) : undefined);

const exported = {
	URL: URLShim,
	URLSearchParams: URLSearchParamsShim,
	parseURL,
};

export default exported;
export { URLShim as URL, URLSearchParamsShim as URLSearchParams, parseURL };
