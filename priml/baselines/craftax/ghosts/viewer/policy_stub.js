'use strict';
// A stand-in for the policy view, the panel under an embed that shows what the
// followed run's agent saw. It keeps the real one's contract, so the embed's
// layout and lockstep can be built and checked before it exists:
//   CraftaxPolicyView.mount(container, { bundleUrl, spritesUrl }) -> Promise<view>
//   view.decisions; view.show(t): the frame before decision t, the final
//   frame from t >= decisions on.
// It draws the decision it was asked for. `decisions` is an extra option the
// embed passes; the real view reads it from its bundle.
(() => {
  window.CraftaxPolicyView = {
    async mount(container, { decisions = Infinity } = {}) {
      const canvas = Object.assign(document.createElement('canvas'), { width: 352, height: 288 });
      canvas.style.cssText = 'display:block;width:min(100%,352px);margin:0 auto;background:#070d11;border-radius:4px';
      canvas.setAttribute('aria-label', 'Placeholder for what the agent sees');
      container.replaceChildren(canvas);
      const ctx = canvas.getContext('2d');
      const view = {
        decisions, shown: -1,
        show(t) {
          view.shown = t;
          const final = t >= decisions;
          ctx.fillStyle = '#070d11';
          ctx.fillRect(0, 0, canvas.width, canvas.height);
          ctx.fillStyle = '#e6edf0';
          ctx.textAlign = 'center';
          ctx.font = '600 15px system-ui, sans-serif';
          ctx.fillText('What the agent sees (placeholder)', 176, 120);
          ctx.font = '600 26px ui-monospace, monospace';
          ctx.fillText(final ? `final frame, ${decisions}` : `decision ${t} of ${decisions}`, 176, 165);
        },
      };
      view.show(0);
      return view;
    },
  };
})();
