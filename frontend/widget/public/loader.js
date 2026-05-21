/**
 * Maintainer's Copilot — widget loader script.
 *
 * Usage:
 *   <script src="https://YOUR_API/widget.js"
 *           data-widget-id="<id>"
 *           data-api-host="https://YOUR_API"
 *           async></script>
 *
 * The script finds its own <script> tag, reads data attributes,
 * then injects a sandboxed iframe pointing at /embed.
 */
(function () {
  var scripts = document.querySelectorAll('script[src*="widget.js"]');
  var scriptTag = scripts[scripts.length - 1];
  if (!scriptTag) return;

  var widgetId = scriptTag.getAttribute("data-widget-id") || "";
  var apiHost = scriptTag.getAttribute("data-api-host") || window.location.origin;

  if (!widgetId) {
    console.warn("[Copilot] data-widget-id is required");
    return;
  }

  var iframe = document.createElement("iframe");
  var params = new URLSearchParams({ widget_id: widgetId, api_host: apiHost });
  iframe.src = apiHost + "/embed?" + params.toString();
  iframe.style.cssText =
    "position:fixed;bottom:0;right:0;width:100%;height:100%;border:none;z-index:2147483647;pointer-events:none;background:transparent";
  iframe.allow = "same-origin";
  iframe.setAttribute(
    "sandbox",
    "allow-scripts allow-same-origin allow-forms allow-popups"
  );
  iframe.setAttribute("title", "Maintainer's Copilot");

  // Enable pointer events only on the bubble / open panel
  // (the iframe itself covers the page; pointer-events pass through via CSS)
  iframe.addEventListener("load", function () {
    iframe.style.pointerEvents = "auto";
  });

  document.body.appendChild(iframe);

  // Respond to resize messages from the widget
  window.addEventListener("message", function (event) {
    if (event.data && event.data.type === "copilot-resize") {
      iframe.style.width = event.data.width || "100%";
      iframe.style.height = event.data.height || "100%";
    }
  });
})();
