export const INTEGRATED_STYLES = `
:host([integrated]) { --ml-text:#1b1f3a; --ml-muted:#68687b; --ml-line:#e7e3dd;
  --ml-line-strong:#d8cbd1; --ml-surface:#fff; --ml-sunken:#fffaf4;
  --ml-accent:#ff5fa2; --ml-accent-ink:#a6295e; --ml-on-accent:#1b1f3a; min-height:0; }
:host([integrated]) .shell { display:block; border:0; border-radius:0; height:100%; min-height:0; }
:host([integrated]) nav { display:none; }
:host([integrated]) main { display:flex; flex-direction:column; height:100%; padding:0; overflow:hidden; }
:host([integrated]) main > div:first-child { flex:1; min-height:0; overflow-y:auto; padding:0 28px 24px; }
:host([integrated]) main header { min-height:77px; margin:0 -28px 20px; padding:16px 28px;
  border-bottom:1px solid var(--ml-line); background:#fff; position:sticky; top:0; z-index:1; align-items:center; }
:host([integrated]) main header h2 { font-size:17px; }
:host([integrated]) .badge { background:#2de2e633; color:#12636a; border:0; }
:host([integrated]) .msg { position:relative; border:0; padding:14px 10px 14px 48px; margin:4px 0; border-radius:10px; }
:host([integrated]) .msg:hover { background:#fffaf4; }
:host([integrated]) .msg::before { content:"●"; display:grid; place-items:center; width:30px; height:30px;
  position:absolute; left:4px; top:12px; border-radius:10px; background:#ffd16655; color:#695012; }
:host([integrated]) .msg pre { line-height:1.7; }
:host([integrated]) .composer { margin:0 28px 18px; padding:12px; border:1px solid #d8cbd1; border-radius:12px; background:#fff; }
:host([integrated]) .composer:focus-within { border-color:#ff5fa2; box-shadow:0 0 0 3px #ff5fa21a; }
:host([integrated]) .composer textarea { background:#fff; border:0; min-height:70px; padding:6px; }
:host([integrated]) .composer input { background:#fffaf4; border:0; font-size:11px; }
:host([integrated]) .composer button { justify-self:end; padding:8px 18px; }
:host([integrated]) .channel-tabs button[aria-pressed=true] { background:#ff5fa226; }
:host([integrated]) .state { padding:30px 10px; }
`;
