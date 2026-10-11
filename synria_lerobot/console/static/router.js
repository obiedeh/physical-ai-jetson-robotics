"use strict";
/** Give every place in the console a hash address so Back, Forward and refresh keep the place.
 *
 * Routes are view selection only. Navigating never opens sources, creates a session or
 * issues a recorder command; those remain explicit operator actions on the screen itself.
 */
(function () {
  const PATTERNS = [
    ["home", /^home$/], ["sessions", /^sessions$/], ["help", /^help$/], ["intro", /^intro$/],
    ["new", /^new(?:\/(task|check|start))?$/],
    ["record", /^session\/([^/]+)\/record$/],
    ["review", /^session\/([^/]+)\/review(?:\/(\d+))?$/],
    ["summary", /^session\/([^/]+)\/summary$/],
  ];

  /** Parse a location hash into a named route, or null for an empty, token or unknown hash. */
  function parse(hash) {
    const raw = String(hash || "").replace(/^#\/?/, "").replace(/\/+$/, "");
    if (!raw || /(^|[&?])token=/.test(raw)) return null;
    for (const [name, pattern] of PATTERNS) {
      const match = raw.match(pattern);
      if (!match) continue;
      if (name === "new") return {name, path: raw, step: match[1] || "task"};
      if (name === "record" || name === "summary") return {name, path: raw, sessionId: decodeURIComponent(match[1])};
      if (name === "review") {
        return {name, path: raw, sessionId: decodeURIComponent(match[1]),
          episode: match[2] === undefined ? null : Number(match[2])};
      }
      return {name, path: raw};
    }
    return null;
  }

  /** Build the hash for a route path such as "sessions" or "session/<id>/record". */
  function href(path) { return "#/" + String(path).replace(/^#?\/?/, ""); }

  /** Address helpers so screens never assemble session routes by hand. */
  const routes = {
    home: () => "home", sessions: () => "sessions", help: () => "help", intro: () => "intro",
    wizard: step => `new/${step || "task"}`,
    record: sessionId => `session/${encodeURIComponent(sessionId)}/record`,
    summary: sessionId => `session/${encodeURIComponent(sessionId)}/summary`,
    review: (sessionId, episode) => `session/${encodeURIComponent(sessionId)}/review`
      + (episode === undefined || episode === null ? "" : `/${episode}`),
  };

  let handler = null, current = null;

  /** Render the route for the current hash, replacing an unknown hash with the fallback. */
  function apply(fallback) {
    const route = parse(location.hash) || parse(href(fallback));
    if (!parse(location.hash)) history.replaceState(null, "", location.pathname + href(route.path));
    current = route;
    if (handler) handler(route);
    return route;
  }

  /** Navigate to a route path; an unchanged hash re-renders because hashchange will not fire. */
  function go(path, options = {}) {
    const target = href(path);
    if (location.hash === target) { apply(path); return; }
    if (options.replace) { history.replaceState(null, "", location.pathname + target); apply(path); return; }
    location.hash = target;
  }

  /** Install the render handler and show the initial place: the hash if valid, else the fallback. */
  function start(render, fallback) {
    handler = render;
    window.addEventListener("hashchange", () => apply(fallback));
    return apply(fallback);
  }

  window.SynriaRouter = {parse, href, routes, go, start, current: () => current};
})();
