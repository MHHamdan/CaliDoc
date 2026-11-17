// CaliDoc WACV 2027 talk — editable deck generator.
// Usage (see README.md): NODE_PATH=<node_modules> node build.js <out.pptx>
// Every number on a slide is traced in ../CaliDoc_WACV2027_Claim_Audit.md.

const path = require("path");
const fs = require("fs");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const lu = require("react-icons/lu");
const NOTES = require("./notes.js");

const OUT = process.argv[2] || path.join(__dirname, "..", "CaliDoc_WACV2027_Presentation.pptx");
const SKILL = process.env.PPTX_SKILL_DIR;
const ASSETS = path.join(__dirname, "assets");

// ---------- theme ----------
const HEX = {
  ink: "1E293B", night: "0F2A2E", white: "FFFFFF", mist: "F1F5F9",
  teal: "0F766E", red: "C2410C", amber: "B45309", slate: "64748B",
  silver: "CBD5E1", tealSoft: "D9EEEC", redSoft: "F9E3DA", amberSoft: "F6E9D7", line: "E2E8F0",
};
const THEME = {
  name: "CaliDoc WACV 2027",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: HEX.ink, lt1: HEX.white, dk2: HEX.night, lt2: HEX.mist,
    accent1: HEX.teal, accent2: HEX.red, accent3: HEX.amber, accent4: HEX.slate,
    accent5: HEX.silver, accent6: "134E4A", hlink: HEX.teal, folHlink: "134E4A",
  },
};

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5 in
pres.author = "Mohammed Hamdan";
pres.company = "Synchromedia Laboratory, ETS Montreal";
pres.title = "CaliDoc: Calibrated Field-Level Risk and Annotation Auditing for Document Information Extraction";
pres.subject = "WACV 2027 talk, paper #1597";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;
const COL = {
  ink: C.text1, white: C.background1, night: C.text2, mist: C.background2,
  teal: C.accent1, red: C.accent2, amber: C.accent3, slate: C.accent4, silver: C.accent5,
};

const W = 13.333, H = 7.5, MX = 0.6;
const FOOT = "CaliDoc  ·  WACV 2027  ·  Paper #1597";

// ---------- layouts ----------
pres.defineSlideMaster({
  title: "TITLE_DARK",
  background: { color: COL.night },
  objects: [],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: COL.white },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.32, w: W - 2 * MX, h: 0.85,
        fontFace: THEME.headFontFace, fontSize: 30, bold: true, color: COL.ink, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { text: { text: FOOT, options: { x: MX, y: 7.08, w: 6, h: 0.28, fontSize: 10, color: COL.slate, margin: 0 } } },
  ],
  slideNumber: { x: 12.23, y: 7.08, w: 0.5, h: 0.28, fontSize: 10, color: COL.slate, align: "right", margin: 0 },
});
pres.defineSlideMaster({
  title: "BACKUP",
  background: { color: COL.white },
  margin: [0.5, 0.6, 0.6, 0.6],
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: MX, y: 0.32, w: W - 2 * MX - 1.3, h: 0.85,
        fontFace: THEME.headFontFace, fontSize: 26, bold: true, color: COL.ink, align: "left", valign: "middle", margin: 0 }, text: "" } },
    { roundRect: { x: W - MX - 1.1, y: 0.52, w: 1.1, h: 0.42, fill: { color: COL.mist }, line: { color: COL.silver, width: 0.75 }, rectRadius: 0.08 } },
    { text: { text: "BACKUP", options: { x: W - MX - 1.1, y: 0.52, w: 1.1, h: 0.42, fontSize: 12, bold: true, color: COL.slate, align: "center", valign: "middle", charSpacing: 2, margin: 0 } } },
    { text: { text: FOOT, options: { x: MX, y: 7.08, w: 6, h: 0.28, fontSize: 10, color: COL.slate, margin: 0 } } },
  ],
  slideNumber: { x: 12.23, y: 7.08, w: 0.5, h: 0.28, fontSize: 10, color: COL.slate, align: "right", margin: 0 },
});

// ---------- helpers ----------
let objN = 0;
const nm = (s) => `${s}_${++objN}`;
// "{_t}" -> subscript t, "{^A}" -> superscript A (plain underscores stay literal, e.g. SUB_TOTAL).
function rich(text, opts = {}) {
  const parts = text.split(/(\{[_^][^}]*\})/).filter((p) => p !== "");
  const runs = parts.map((p, i) => {
    const m = p.match(/^\{([_^])([^}]*)\}$/);
    const o = Object.assign({}, opts); delete o.breakLine;
    // paragraph-level options only on the first run, or pptxgenjs starts a new paragraph per run
    if (i > 0) ["bullet", "align", "indentLevel", "paraSpaceAfter", "paraSpaceBefore"].forEach((k) => delete o[k]);
    if (m) o[m[1] === "_" ? "subscript" : "superscript"] = true;
    return { text: m ? m[2] : p, options: o };
  });
  if (opts.breakLine && runs.length) runs[runs.length - 1].options.breakLine = true;
  return runs;
}
function T(slide, text, o) {
  text = typeof text === "string" ? (/\{[_^]/.test(text) ? rich(text) : text) : text.flatMap((r) => rich(r.text, r.options || {}));
  slide.addText(text, Object.assign({ isTextBox: true, margin: 0, color: COL.ink, fontSize: 16, valign: "top", objectName: nm("text") }, o));
}
function box(slide, x, y, w, h, o = {}) {
  slide.addShape(o.round === false ? pres.shapes.RECTANGLE : pres.shapes.ROUNDED_RECTANGLE, Object.assign({
    x, y, w, h, rectRadius: o.round === false ? undefined : (o.r ?? 0.08),
    fill: { color: o.fill ?? COL.mist, transparency: o.ft ?? 0 },
    line: o.line ? { color: o.line, width: o.lw ?? 1, dashType: o.dash } : { type: "none" },
    objectName: nm(o.name || "shape"),
  }, o.extra || {}));
}
function rule(slide, x1, y1, x2, y2, color, width = 1.5, o = {}) {
  slide.addShape(pres.shapes.LINE, {
    x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.0001, h: Math.abs(y2 - y1) || 0.0001,
    flipH: false, line: { color, width, dashType: o.dash, endArrowType: o.arrow ? "triangle" : undefined, beginArrowType: o.barrow ? "triangle" : undefined },
    objectName: nm("line"),
  });
}
function arrow(slide, x1, y1, x2, y2, color = COL.slate, width = 2) { rule(slide, x1, y1, x2, y2, color, width, { arrow: true }); }
function source(slide, text) {
  T(slide, text, { x: MX, y: 6.66, w: W - 2 * MX, h: 0.3, fontSize: 11, italic: true, color: COL.slate, valign: "bottom" });
}
function title(slide, text) { slide.addText(text, { placeholder: "title" }); }
// Equation image sized so its glyphs render at `pt` points (images are rendered at 30 pt, 300 dpi).
const { imageSize } = (() => {
  const sz = (f) => { const b = fs.readFileSync(f); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; };
  return { imageSize: sz };
})();
function eq(slide, name, x, y, pt = 22, alt) {
  const f = path.join(ASSETS, name + ".png");
  const { w, h } = imageSize(f);
  const k = pt / 30 / 300;
  slide.addImage({ path: f, x, y, w: w * k, h: h * k, altText: alt || name, objectName: nm("equation") });
  return { w: w * k, h: h * k };
}
async function icon(Comp, hex, px = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(Comp, { color: "#" + hex, size: px }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}
function chip(slide, x, y, w, h, kind) {
  const m = { ok: [COL.teal, 30, COL.teal], bad: [COL.red, 0, COL.red], neutral: [COL.silver, 40, COL.slate], edge: [COL.red, 0, COL.red], ghost: [COL.white, 0, COL.silver] }[kind];
  box(slide, x, y, w, h, { fill: m[0], ft: m[1], line: m[2], lw: 0.75, r: 0.05, name: "token" });
}
function notesText(n) {
  const mm = Math.floor(n.time / 60), ss = String(n.time % 60).padStart(2, "0");
  const parts = [];
  parts.push(n.backup ? "[BACKUP — Q&A only]" : `[~${mm}:${ss}]  ${n.title}`);
  parts.push("");
  parts.push(n.say.join("\n\n"));
  if (n.emphasize) parts.push("\nEMPHASIZE: " + n.emphasize);
  if (n.why) parts.push("WHY IT MATTERS: " + n.why);
  if (n.transition) parts.push("TRANSITION: " + n.transition);
  if (n.avoid) parts.push("AVOID: " + n.avoid);
  if (n.sources) parts.push("SOURCES: " + n.sources);
  return parts.join("\n");
}
// Two pptxgenjs output fixes, applied to the written package:
// 1. combo charts reference a third axis id that is never declared; LibreOffice then
//    drops the chart. Remove every <c:axId> that no axis element declares.
// 2. a paragraph with several runs gets one <a:pPr> per run (schema allows one, first);
//    keep only the first, which carries the bullet and spacing.
async function fixPackage(file) {
  const JSZip = require("jszip");
  const zip = await JSZip.loadAsync(fs.readFileSync(file));
  for (const name of Object.keys(zip.files).filter((n) => /^ppt\/slides\/slide\d+\.xml$/.test(n))) {
    let xml = await zip.file(name).async("string");
    xml = xml.replace(/<a:p>([\s\S]*?)<\/a:p>/g, (m, body) => {
      let seen = false;
      body = body.replace(/<a:pPr\b[^>]*\/>|<a:pPr\b[^>]*>[\s\S]*?<\/a:pPr>/g, (pp) => (seen ? "" : ((seen = true), pp)));
      return "<a:p>" + body + "</a:p>";
    });
    zip.file(name, xml);
  }
  for (const name of Object.keys(zip.files).filter((n) => /^ppt\/charts\/chart\d+\.xml$/.test(n))) {
    let xml = await zip.file(name).async("string");
    const declared = new Set([...xml.matchAll(/<c:(?:catAx|valAx|dateAx|serAx)>\s*<c:axId val="(\d+)"\/>/g)].map((m) => m[1]));
    xml = xml.replace(/<c:axId val="(\d+)"\/>/g, (m, id) => (declared.has(id) ? m : ""));
    zip.file(name, xml);
  }
  fs.writeFileSync(file, await zip.generateAsync({ type: "nodebuffer", compression: "DEFLATE" }));
}
const byId = Object.fromEntries(NOTES.map((n) => [n.id, n]));
function addNotes(slide, id) { slide.addNotes(notesText(byId[id])); }

const AX = { // shared chart styling
  catAxisLabelColor: HEX.ink, valAxisLabelColor: HEX.slate, catAxisLabelFontSize: 14, valAxisLabelFontSize: 12,
  catAxisLabelFontFace: "Calibri", valAxisLabelFontFace: "Calibri", dataLabelFontFace: "Calibri",
  valGridLine: { color: HEX.line, size: 0.75 }, catGridLine: { style: "none" },
  catAxisLineShow: true, catAxisLineColor: HEX.silver, valAxisLineShow: false,
  legendFontFace: "Calibri", legendFontSize: 13, legendColor: HEX.ink,
  dataLabelFontSize: 12, dataLabelColor: HEX.ink,
};

(async () => {
  const ic = {
    risk: await icon(lu.LuSigma, HEX.teal), layers: await icon(lu.LuLayers, HEX.teal), ruler: await icon(lu.LuRuler, HEX.teal),
    link: await icon(lu.LuLink2, HEX.teal), target: await icon(lu.LuTarget, HEX.teal), globe: await icon(lu.LuGlobe, HEX.teal),
    receipt: await icon(lu.LuReceipt, HEX.slate), db: await icon(lu.LuDatabase, HEX.slate), sheet: await icon(lu.LuFileSpreadsheet, HEX.slate),
    shield: await icon(lu.LuShieldCheck, HEX.teal), scan: await icon(lu.LuScanSearch, HEX.amber),
    gh: await icon(lu.LuGithub, "E2E8F0"), book: await icon(lu.LuBookOpen, "E2E8F0"), play: await icon(lu.LuPlay, "E2E8F0"), box: await icon(lu.LuBox, "E2E8F0"),
  };

  // =====================================================================
  // 1 — Title
  // =====================================================================
  pres.addSection({ title: "Talk" });
  {
    const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Talk" });
    // motif: a row of token chips, all covered except a field whose edge escapes
    const y0 = 1.05;
    const kinds = ["ghost", "ghost", "ok", "ok", "ok", "ok", "ok", "ok", "bad", "ghost", "ghost"];
    kinds.forEach((k, i) => box(s, MX + i * 0.42, y0, 0.34, 0.2, {
      fill: k === "ok" ? COL.teal : k === "bad" ? COL.red : COL.white, ft: k === "ghost" ? 85 : 0, r: 0.04, name: "motif" }));
    T(s, "WACV 2027  ·  Evaluations & Datasets Track  ·  Paper #1597", { x: MX, y: 1.55, w: 11, h: 0.4, fontSize: 16, color: "9FD3CC", bold: true, charSpacing: 1 });
    T(s, [
      { text: "CaliDoc", options: { bold: true, color: COL.white } },
      { text: ": Calibrated Field-Level Risk and Annotation Auditing for Document Information Extraction", options: { color: COL.white } },
    ], { x: MX, y: 2.05, w: 11.6, h: 2.0, fontFace: THEME.headFontFace, fontSize: 38, valign: "top", lineSpacingMultiple: 0.95 });
    T(s, "Mohammed Hamdan    Mohamed Cheriet", { x: MX, y: 4.55, w: 11, h: 0.45, fontSize: 22, bold: true, color: COL.white });
    T(s, "Synchromedia Laboratory, École de technologie supérieure (ÉTS), Montréal, Canada", { x: MX, y: 5.05, w: 11, h: 0.4, fontSize: 18, color: "CBD5E1" });
    T(s, "Token-level reliability is not complete-field reliability.", { x: MX, y: 6.35, w: 11, h: 0.45, fontSize: 18, italic: true, color: "9FD3CC" });
    addNotes(s, 1);
  }

  // =====================================================================
  // 2 — Motivation
  // =====================================================================
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
    title(s, "Extraction pipelines consume complete fields, not tokens");
    // conceptual receipt
    const rx = MX, ry = 1.55, rw = 2.9, rh = 3.75;
    box(s, rx, ry, rw, rh, { fill: COL.white, line: COL.silver, lw: 1.25, r: 0.06, name: "receipt" });
    const fields = [["COMPANY", 0.25, 0.5], ["ADDRESS", 0.9, 0.85], ["DATE", 2.0, 0.32], ["TOTAL", 3.05, 0.32]];
    for (let k = 0; k < 11; k++) { const yy = ry + 0.45 + k * 0.29; box(s, rx + 0.3, yy, rw - 0.6 - (k % 3) * 0.35, 0.07, { fill: COL.silver, ft: 30, round: false, name: "ocrline" }); }
    fields.forEach(([n, dy, hh]) => { box(s, rx + 0.18, ry + dy, rw - 0.36, hh, { fill: COL.teal, ft: 88, line: COL.teal, lw: 1.25, r: 0.04, name: "fieldbox" }); });
    T(s, "Conceptual illustration", { x: rx, y: ry + rh + 0.08, w: rw, h: 0.3, fontSize: 11, italic: true, color: COL.slate, align: "center" });
    arrow(s, 3.7, 3.4, 4.4, 3.4, COL.slate, 2.25);
    // fields list
    const fx = 4.6;
    T(s, "Delivered output", { x: fx, y: 1.55, w: 3, h: 0.35, fontSize: 14, bold: true, color: COL.slate });
    fields.forEach(([n], i) => {
      box(s, fx, 2.0 + i * 0.72, 2.7, 0.55, { fill: COL.teal, ft: 88, line: COL.teal, r: 0.06, name: "field" });
      T(s, n, { x: fx + 0.15, y: 2.0 + i * 0.72, w: 2.4, h: 0.55, fontSize: 18, bold: true, color: COL.teal, valign: "middle" });
    });
    arrow(s, 7.5, 3.4, 8.2, 3.4, COL.slate, 2.25);
    // downstream
    const dx = 8.4;
    T(s, "Downstream decisions", { x: dx, y: 1.55, w: 4.3, h: 0.35, fontSize: 14, bold: true, color: COL.slate });
    [[ic.sheet, "Expense & accounting systems"], [ic.db, "Records & databases"], [ic.receipt, "Compliance & audit"]].forEach(([im, t], i) => {
      s.addImage({ data: im, x: dx, y: 2.08 + i * 0.95, w: 0.5, h: 0.5, altText: t, objectName: nm("icon") });
      T(s, t, { x: dx + 0.7, y: 2.05 + i * 0.95, w: 3.7, h: 0.56, fontSize: 18, valign: "middle" });
    });
    // the key sentence + span illustration
    box(s, MX, 5.62, W - 2 * MX, 0.95, { fill: COL.mist, r: 0.08, name: "callout" });
    T(s, [
      { text: "A field is correct only if every word in its span is correct.", options: { bold: true } },
      { text: "   Conformal prediction for KIE has so far certified tokens or one representative token per entity.", options: { color: COL.slate } },
    ], { x: MX + 0.25, y: 5.62, w: 8.1, h: 0.95, fontSize: 16, valign: "middle" });
    for (let i = 0; i < 11; i++) chip(s, 9.05 + i * 0.3, 5.92, 0.24, 0.34, i === 10 ? "bad" : "ok");
    T(s, "1 wrong word → wrong ADDRESS", { x: 9.05, y: 6.28, w: 3.4, h: 0.26, fontSize: 12, color: COL.red, bold: true });
    source(s, "Main paper Sec. 1–2.2");
    addNotes(s, 2);
  }

  // =====================================================================
  // 3 — The SROIE counterexample
  // =====================================================================
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
    title(s, "89.8% token coverage, yet 0 of 420 fields fully covered");
    T(s, "SROIE  ·  LayoutLMv3-Large (entity F1 94.54 ± 0.33)  ·  token-marginal LAC conformal prediction at the 90% target",
      { x: MX, y: 1.2, w: W - 2 * MX, h: 0.35, fontSize: 15, color: COL.slate });
    // left: tokens (each cell ~1% of tokens)
    const lx = MX, ly = 1.85;
    box(s, lx, ly, 5.55, 4.0, { fill: COL.teal, ft: 92, r: 0.08, name: "panel" });
    T(s, "89.8%", { x: lx + 0.3, y: ly + 0.15, w: 2.6, h: 1.0, fontSize: 60, bold: true, color: COL.teal, fontFace: THEME.headFontFace });
    T(s, "of tokens covered\n(target 90%)", { x: lx + 0.3, y: ly + 1.2, w: 2.4, h: 0.8, fontSize: 18, color: COL.ink });
    for (let i = 0; i < 100; i++) {
      const r = Math.floor(i / 10), c = i % 10;
      box(s, lx + 2.95 + c * 0.235, ly + 0.25 + r * 0.235, 0.19, 0.19, { fill: i < 90 ? COL.teal : COL.white, ft: 0, line: i < 90 ? COL.teal : COL.silver, lw: 0.75, round: false, name: "tokcell" });
    }
    T(s, "1 cell ≈ 1% of tokens", { x: lx + 2.95, y: ly + 2.65, w: 2.4, h: 0.3, fontSize: 12, color: COL.slate });
    T(s, "Looks calibrated.", { x: lx + 0.3, y: ly + 3.25, w: 5, h: 0.5, fontSize: 20, italic: true, color: COL.teal, bold: true });
    // right: fields
    const rx = 6.55, ry = 1.85;
    box(s, rx, ry, 6.18, 4.0, { fill: COL.red, ft: 92, r: 0.08, name: "panel" });
    T(s, "0 / 420", { x: rx + 0.3, y: ry + 0.15, w: 3.4, h: 1.0, fontSize: 60, bold: true, color: COL.red, fontFace: THEME.headFontFace });
    T(s, "gold field spans with every token covered  (full-span coverage 0.000)", { x: rx + 3.55, y: ry + 0.3, w: 2.45, h: 0.95, fontSize: 15, color: COL.ink });
    for (let i = 0; i < 420; i++) {
      const r = Math.floor(i / 30), c = i % 30;
      box(s, rx + 0.32 + c * 0.185, ry + 1.42 + r * 0.155, 0.14, 0.115, { fill: COL.white, line: COL.red, lw: 0.5, round: false, name: "fieldcell" });
    }
    T(s, "1 outline = 1 gold field; none is fully covered", { x: rx + 0.32, y: ry + 3.62, w: 5.6, h: 0.28, fontSize: 12, color: COL.slate });
    // mechanism
    T(s, [
      { text: "Why: ", options: { bold: true, color: COL.red } },
      { text: "one marginal threshold (0.0056) is sized for the O class (84% of tokens). 6 of 9 classes, including all four B- classes, get 0% coverage — and every field begins with a B- token." },
    ], { x: MX, y: 6.0, w: W - 2 * MX, h: 0.6, fontSize: 16, valign: "middle" });
    source(s, "Main paper Sec. 4.4.2, Table 1 (mean over 3 calibration splits); Supp. Table S6");
    addNotes(s, 3);
  }

  // =====================================================================
  // 4 — Tokens vs fields on a real receipt
  // =====================================================================
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Talk" });
    title(s, "Change the calibrated object: tokens → complete fields");
    T(s, "One real SROIE test receipt, released checkpoint. Text withheld (dataset licence unclear); structure and scores are genuine.",
      { x: MX, y: 1.2, w: W - 2 * MX, h: 0.35, fontSize: 15, color: COL.slate });
    const rows = [ // type, words, worst-word score, field threshold (Supp. Table S13)
      ["COMPANY", 4, 0.04417, 0.1402], ["ADDRESS", 11, 0.02887, 0.1013], ["DATE", 1, 0.00787, 0.0224], ["TOTAL", 1, 0.01132, 0.0765],
    ];
    const ax0 = 6.25, aw = 4.0, lg = (v) => ax0 + (Math.log10(v) + 3) / 3 * aw; // 1e-3 .. 1
    const y0 = 2.35, dy = 0.92;
    T(s, "Field", { x: MX, y: 1.8, w: 1.6, h: 0.3, fontSize: 13, bold: true, color: COL.slate });
    T(s, "Words in span", { x: 2.3, y: 1.8, w: 2.5, h: 0.3, fontSize: 13, bold: true, color: COL.slate });
    T(s, "Worst-word score  1 − p(label)   (log scale)", { x: ax0, y: 1.8, w: aw, h: 0.3, fontSize: 13, bold: true, color: COL.slate });
    T(s, "Token-marginal", { x: 10.4, y: 1.8, w: 1.4, h: 0.3, fontSize: 13, bold: true, color: COL.red });
    T(s, "CaliDoc", { x: 11.85, y: 1.8, w: 0.9, h: 0.3, fontSize: 13, bold: true, color: COL.teal });
    rows.forEach(([n, k, sc, lam], i) => {
      const y = y0 + i * dy;
      if (i % 2 === 0) box(s, MX - 0.1, y - 0.22, W - 2 * MX + 0.2, dy - 0.06, { fill: COL.mist, ft: 30, r: 0.04, name: "rowband" });
      T(s, n, { x: MX, y: y - 0.1, w: 1.6, h: 0.42, fontSize: 18, bold: true, valign: "middle" });
      for (let j = 0; j < k; j++) chip(s, 2.3 + j * 0.33, y - 0.05, 0.27, 0.32, "neutral");
      // field threshold tick
      rule(s, lg(lam), y - 0.16, lg(lam), y + 0.43, COL.teal, 3);
      T(s, lam.toFixed(3), { x: lg(lam) + 0.06, y: y - 0.2, w: 0.6, h: 0.25, fontSize: 11, color: COL.teal, bold: true });
      s.addShape(pres.shapes.OVAL, { x: lg(sc) - 0.11, y: y + 0.03, w: 0.22, h: 0.22, fill: { color: COL.ink }, line: { color: COL.white, width: 1 }, objectName: nm("dot") });
      T(s, sc.toFixed(4), { x: lg(sc) - 0.3, y: y + 0.27, w: 0.6, h: 0.2, fontSize: 11, color: COL.ink, align: "center" });
      T(s, "missed", { x: 10.4, y: y - 0.05, w: 1.15, h: 0.36, fontSize: 16, bold: true, color: COL.red, valign: "middle" });
      T(s, "covered", { x: 11.85, y: y - 0.05, w: 0.95, h: 0.36, fontSize: 16, bold: true, color: COL.teal, valign: "middle" });
    });
    // marginal threshold line across all rows
    const qx = lg(0.0056), yb = y0 + 3 * dy + 0.5;
    rule(s, qx, y0 - 0.28, qx, yb, COL.red, 2.25, { dash: "dash" });
    // axis
    rule(s, ax0, yb, ax0 + aw, yb, COL.slate, 1);
    [0.001, 0.01, 0.1, 1].forEach((v) => { rule(s, lg(v), yb, lg(v), yb + 0.08, COL.slate, 1); T(s, String(v), { x: lg(v) - 0.35, y: yb + 0.1, w: 0.7, h: 0.25, fontSize: 11, color: COL.slate, align: "center" }); });
    // legend
    const ly = 6.08;
    rule(s, MX, ly + 0.15, MX + 0.4, ly + 0.15, COL.red, 2.25, { dash: "dash" });
    T(s, "Token-marginal threshold 0.0056", { x: MX + 0.5, y: ly, w: 3.6, h: 0.3, fontSize: 14, valign: "middle", bold: true });
    rule(s, 4.6, ly, 4.6, ly + 0.3, COL.teal, 3);
    T(s, "CaliDoc field threshold (SpanCRC, α = 0.10)", { x: 4.75, y: ly, w: 4.4, h: 0.3, fontSize: 14, valign: "middle", bold: true });
    T(s, "● worst word of the field", { x: 9.6, y: ly, w: 3.1, h: 0.3, fontSize: 14, valign: "middle", bold: true });
    source(s, "Released checkpoint MHamdan/calidoc-layoutlmv3-large-sroie, receipt sroie_test_X51005568889; thresholds: main Table 1, Supp. Table S13. Illustrative single document.");
    addNotes(s, 4);
  }

  // =====================================================================
  // 5 — Framework
  // =====================================================================
  pres.addSection({ title: "Method" });
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
    title(s, "CaliDoc: one calibrated threshold, two uses");
    const y = 1.45, h = 1.05, bw = 2.65, gap = 0.4;
    const top = [
      ["Document + OCR", "image, words, boxes", false],
      ["LayoutLMv3-Large tagger", "fixed · argmax BIO decoding", false],
      ["Word probabilities", "temperature fit on early-stopping split", false],
      ["Span score S(s)", "worst word in each field span", true],
    ];
    top.forEach(([a, b, hi], i) => {
      const x = MX + i * (bw + gap);
      box(s, x, y, bw, h, { fill: hi ? COL.teal : COL.mist, ft: hi ? 88 : 0, line: hi ? COL.teal : COL.slate, lw: 1.25, name: "stage" });
      T(s, a, { x: x + 0.12, y: y + 0.08, w: bw - 0.24, h: 0.45, fontSize: 17, bold: true, align: "center", valign: "middle" });
      T(s, b, { x: x + 0.12, y: y + 0.53, w: bw - 0.24, h: 0.45, fontSize: 13, color: COL.slate, align: "center", valign: "middle" });
      if (i < 3) arrow(s, x + bw + 0.04, y + h / 2, x + bw + gap - 0.04, y + h / 2);
    });
    T(s, "no retraining · no constrained decoding · no sequence search", { x: MX, y: y + h + 0.1, w: 3 * bw + 2 * gap, h: 0.3, fontSize: 13, italic: true, color: COL.slate, align: "center" });
    // one connector from the span score back to the left edge, feeding both lanes
    const sx = MX + 3 * (bw + gap) + bw / 2, hy = 3.1, vx = MX + 0.12, x0 = MX + 0.5, step = 3.92, ww = 3.5;
    const lanes = [
      { y: 3.6, col: COL.teal, label: "CALIBRATION PATH", steps: [
        ["Document-level risk control", "documents are the exchangeable unit"],
        ["Thresholds", "pooled λ, plus λ{_t} per supported field type (m0 = 25)"],
        ["Per-field calibrated sets", "E[fraction of fields missed per document] ≤ α"] ] },
      { y: 5.2, col: COL.amber, label: "AUDIT PATH  ·  reuses the same threshold", steps: [
        ["Escapes inside spans", "reference τ = pooled λ (no new parameter)"],
        ["Boundary-escape test", "edge statistic vs conditional-randomization null"],
        ["Field-level finding", "Holm + document-blocked bootstrap"] ] },
    ];
    rule(s, sx, y + h, sx, hy, COL.slate, 2);
    rule(s, vx, hy, sx, hy, COL.slate, 2);
    rule(s, vx, hy, vx, lanes[1].y + 0.52, COL.slate, 2);
    lanes.forEach((ln) => {
      T(s, ln.label, { x: x0, y: ln.y - 0.36, w: 6, h: 0.3, fontSize: 13, bold: true, color: ln.col, charSpacing: 1 });
      arrow(s, vx, ln.y + 0.52, x0 - 0.04, ln.y + 0.52, ln.col, 2);
      ln.steps.forEach(([a, b], i) => {
        const x = x0 + i * step;
        box(s, x, ln.y, ww, 1.05, { fill: ln.col, ft: 90, line: ln.col, lw: 1.25, name: "lane" });
        T(s, a, { x: x + 0.14, y: ln.y + 0.07, w: ww - 0.28, h: 0.42, fontSize: 17, bold: true, color: ln.col, valign: "middle" });
        T(s, b, { x: x + 0.14, y: ln.y + 0.49, w: ww - 0.28, h: 0.5, fontSize: 13.5, color: COL.ink, valign: "middle" });
        if (i < 2) arrow(s, x + ww + 0.04, ln.y + 0.52, x + step - 0.04, ln.y + 0.52, ln.col);
      });
    });
    source(s, "Main paper Sec. 3; temperature fitting Sec. 3.1, Supp. S1");
    addNotes(s, 5);
  }

  // =====================================================================
  // 6 — Span score + document-level CRC
  // =====================================================================
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
    title(s, "Score the worst word, calibrate over documents");
    // left panel: span score
    const lx = MX, ly = 1.4;
    T(s, "1  Score each gold span by its worst word", { x: lx, y: ly, w: 5.6, h: 0.4, fontSize: 18, bold: true, color: COL.teal });
    eq(s, "eq_span_score", lx, ly + 0.55, 22, "S(s) = max over a<=i<b of (1 - p(g_i | x_i))");
    // bar diagram (conceptual)
    const bx = lx + 0.2, by = 4.6, scores = [0.25, 0.4, 0.3, 1.15, 0.35];
    scores.forEach((v, i) => {
      const hh = v * 1.4;
      box(s, bx + i * 0.82, by - hh, 0.55, hh, { fill: i === 3 ? COL.red : COL.teal, ft: i === 3 ? 0 : 35, round: false, name: "scorebar" });
      chip(s, bx + i * 0.82, by + 0.08, 0.55, 0.3, i === 3 ? "bad" : "ok");
    });
    const lamY = by - 0.95;
    rule(s, bx - 0.15, lamY, bx + 4.2, lamY, COL.ink, 2, { dash: "dash" });
    T(s, "λ", { x: bx + 4.25, y: lamY - 0.2, w: 0.4, h: 0.4, fontSize: 20, bold: true, valign: "middle" });
    T(s, "S(s) = tallest bar", { x: bx + 2.0, y: by - 1.9, w: 2.4, h: 0.3, fontSize: 13, color: COL.red, bold: true });
    T(s, [
      { text: "Span fully covered at λ  iff  S(s) ≤ λ", options: { bold: true, breakLine: true } },
      { text: "The max is forced by this event: a mean or quantile certifies a different event. Price: conservatism on long spans.", options: { color: COL.slate } },
    ], { x: lx, y: by + 0.55, w: 5.5, h: 1.25, fontSize: 15 });
    T(s, "Conceptual bars", { x: bx, y: by + 0.4, w: 2, h: 0.2, fontSize: 10, italic: true, color: COL.slate });
    // right panel: CRC
    const rx = 6.75;
    T(s, "2  Calibrate over documents, not spans", { x: rx, y: ly, w: 6, h: 0.4, fontSize: 18, bold: true, color: COL.teal });
    T(s, "Fraction of document j's gold fields missed:", { x: rx, y: ly + 0.5, w: 6, h: 0.3, fontSize: 14, color: COL.slate });
    eq(s, "eq_doc_risk", rx, ly + 0.85, 21, "R_j(lambda) = (1/|S_j|) sum over s in S_j of 1[S(s) > lambda]");
    T(s, "Smallest λ whose corrected calibration risk ≤ α:", { x: rx, y: ly + 1.75, w: 6, h: 0.3, fontSize: 14, color: COL.slate });
    eq(s, "eq_lambda_hat", rx, ly + 2.1, 21, "lambda-hat = inf { lambda : (n R-hat(lambda) + 1)/(n+1) <= alpha }");
    // guarantee box
    const gy = 4.35;
    box(s, rx, gy, 5.98, 1.1, { fill: COL.teal, ft: 88, line: COL.teal, lw: 1.5, name: "guarantee" });
    eq(s, "eq_guarantee", rx + 0.2, gy + 0.33, 21, "E[R_{n+1}(lambda-hat)] <= alpha");
    T(s, "On a fresh exchangeable document with ≥ 1 gold field, the expected fraction of gold fields missed is at most α.",
      { x: rx + 2.5, y: gy + 0.08, w: 3.3, h: 0.95, fontSize: 13.5, valign: "middle", bold: true, color: COL.ink });
    box(s, rx, gy + 1.22, 5.98, 0.95, { fill: COL.red, ft: 92, r: 0.06, name: "notbox" });
    T(s, [
      { text: "Not guaranteed: ", options: { bold: true, color: COL.red } },
      { text: "that each document misses ≤ α of its fields; that each field type is covered at 1 − α; that a split's empirical full-span coverage reaches 1 − α." },
    ], { x: rx + 0.15, y: gy + 1.22, w: 5.7, h: 0.95, fontSize: 13.5, valign: "middle" });
    source(s, "Main paper Sec. 3.3, Eqs. 1–3; Sec. 5.2. Assumes calibration and test documents are exchangeable; calibration disjoint from training, temperature and model selection.");
    addNotes(s, 6);
  }

  // =====================================================================
  // 7 — Field-conditional thresholds
  // =====================================================================
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
    title(s, "Per-field thresholds where supported, pooled elsewhere");
    const lx = MX;
    T(s, "Deployed candidate-label rule", { x: lx, y: 1.4, w: 5.6, h: 0.35, fontSize: 16, bold: true, color: COL.teal });
    eq(s, "eq_candidate", lx, 1.85, 20, "C(x_i) = { y : 1 - p(y | x_i) <= lambda-hat_type(y) }");
    T(s, [
      { text: "Same CRC per field type t → threshold λ{_t}", options: { bullet: true, breakLine: true } },
      { text: "Gold type-t span covered iff S(s) ≤ λ{_t}: deployed rule = calibrated event", options: { bullet: true, breakLine: true } },
      { text: "Fewer than m0 = 25 calibration spans → pooled λ", options: { bullet: true, bold: true } },
    ], { x: lx, y: 2.5, w: 5.75, h: 1.75, fontSize: 15, paraSpaceAfter: 6 });
    // formal box
    box(s, lx, 4.45, 5.7, 2.0, { fill: COL.mist, line: COL.silver, r: 0.06, name: "formal" });
    T(s, "What is formal", { x: lx + 0.2, y: 4.53, w: 5.3, h: 0.35, fontSize: 15, bold: true, color: COL.teal });
    T(s, [
      { text: "Pooled document-level bound", options: { bullet: true, breakLine: true } },
      { text: "Per-type bound for each supported type", options: { bullet: true, breakLine: true } },
      { text: "Fallback types inherit only the pooled bound", options: { bullet: true, breakLine: true, bold: true } },
      { text: "Aggregate full-span coverage under the mixed rule is empirical", options: { bullet: true, bold: true } },
    ], { x: lx + 0.2, y: 4.9, w: 5.35, h: 1.5, fontSize: 14, paraSpaceAfter: 2 });
    // chart: CORD calibration span counts
    const cord = [ ["MENU.CNT",108],["MENU.NM",108],["MENU.PRICE",108],["TOTAL.TOTAL_PRICE",49],["MENU.SUB.NM",41],["TOTAL.CHANGEPRICE",34],["SUB_TOTAL.SUBTOTAL_PRICE",33],["MENU.SUB.CNT",31],["TOTAL.CASHPRICE",31],
      ["SUB_TOTAL.TAX_PRICE",21],["MENU.UNITPRICE",19],["TOTAL.MENUQTY_CNT",13],["MENU.SUB.PRICE",9],["TOTAL.CREDITCARDPRICE",6],["SUB_TOTAL.SERVICE_PRICE",5],["MENU.NUM",2],["SUB_TOTAL.DISCOUNT_PRICE",2],["SUB_TOTAL.ETC",2],["TOTAL.EMONEYPRICE",2],["TOTAL.TOTAL_ETC",2],["TOTAL.MENUTYPE_CNT",1],["MENU.DISCOUNTPRICE",0],["MENU.ETC",0],["MENU.ITEMSUBTOTAL",0] ];
    // Supp. Table S14 lists 24 CORD field types for the first split (Supp. S4.3 prose says 25; see claim audit).
    const labels = cord.map((_, i) => String(i + 1));
    T(s, "CORD: calibration spans per field type (Supp. Table S14)", { x: 6.75, y: 1.4, w: 6, h: 0.35, fontSize: 16, bold: true, color: COL.ink });
    s.addChart([
      { type: pres.charts.BAR, data: [
        { name: "Supported (own threshold)", labels, values: cord.map(([, v]) => (v >= 25 ? v : 0)) },
        { name: "Pooled fallback", labels, values: cord.map(([, v]) => (v < 25 ? v : 0)) },
      ], options: { barGrouping: "stacked", chartColors: [HEX.teal, HEX.silver] } },
      { type: pres.charts.LINE, data: [{ name: "m0 = 25", labels, values: cord.map(() => 25) }], options: { chartColors: [HEX.red], lineSize: 2, lineDataSymbol: "none", lineDash: "dash", showValue: false } },
    ], Object.assign({}, AX, {
      x: 6.65, y: 1.8, w: 6.1, h: 3.45, barDir: "col", barGapWidthPct: 35,
      showLegend: true, legendPos: "t", catAxisLabelFontSize: 9, catAxisHidden: true,
      valAxisMaxVal: 120, valAxisMinVal: 0, valAxisMajorUnit: 25, showValue: false,
      valAxisTitle: "calibration spans", showValAxisTitle: true, valAxisTitleFontSize: 12, valAxisTitleColor: HEX.slate, valAxisTitleFontFace: "Calibri",
      objectName: nm("chart_cord_support"),
    }));
    T(s, "Field types, sorted by support", { x: 6.75, y: 5.22, w: 5.9, h: 0.25, fontSize: 11, color: COL.slate, align: "center" });
    box(s, 6.75, 5.55, 5.98, 0.9, { fill: COL.amber, ft: 90, r: 0.06, name: "cordnote" });
    T(s, [
      { text: "SROIE: ", options: { bold: true } }, { text: "all 4 field types supported (242 calibration spans each).", options: { breakLine: true } },
      { text: "CORD: ", options: { bold: true } }, { text: "most types fall back (9 of 24 listed in Table S14 supported); 19.3% of evaluation spans fall in fallback types." },
    ], { x: 6.95, y: 5.55, w: 5.7, h: 0.9, fontSize: 14, valign: "middle" });
    source(s, "Main paper Sec. 3.3; Supp. S4, Tables S13–S14");
    addNotes(s, 7);
    if (cord.length !== 24) throw new Error("CORD field list length " + cord.length);
  }

  // =====================================================================
  // 8 — Main results
  // =====================================================================
  pres.addSection({ title: "Results" });
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
    title(s, "SpanCRC repairs SROIE without disturbing working datasets");
    const ds = ["SROIE", "SROIE-rep.", "WildReceipt", "FUNSD", "CORD"];
    s.addChart([
      { type: pres.charts.BAR, data: [
        { name: "Token-marginal threshold", labels: ds, values: [0.000, 0.257, 0.918, 0.950, 0.872] },
        { name: "Field-conditional SpanCRC", labels: ds, values: [0.909, 0.894, 0.913, 0.944, 0.886] },
      ], options: { barGrouping: "clustered", chartColors: [HEX.silver, HEX.teal] } },
      { type: pres.charts.LINE, data: [{ name: "Target 0.90", labels: ds, values: ds.map(() => 0.9) }], options: { chartColors: [HEX.red], lineSize: 1.5, lineDataSymbol: "none", lineDash: "dash", showValue: false } },
    ], Object.assign({}, AX, {
      x: MX - 0.1, y: 1.35, w: 6.6, h: 4.75, barDir: "col", barGapWidthPct: 55,
      showLegend: true, legendPos: "t", valAxisMaxVal: 1, valAxisMinVal: 0, valAxisMajorUnit: 0.25, valAxisLabelFormatCode: "0.00",
      showValue: true, dataLabelFormatCode: "0.000", dataLabelPosition: "outEnd", dataLabelFontSize: 11,
      valAxisTitle: "Empirical full-span coverage", showValAxisTitle: true, valAxisTitleFontSize: 13, valAxisTitleColor: HEX.slate, valAxisTitleFontFace: "Calibri",
      objectName: nm("chart_fullspan"),
    }));
    // compact table
    const hdr = (t) => ({ text: t, options: { bold: true, color: HEX.slate, fontSize: 13, fill: { color: HEX.mist }, valign: "middle" } });
    const cell = (t, o = {}) => ({ text: t, options: Object.assign({ fontSize: 15, color: HEX.ink, valign: "middle" }, o) });
    const rows = [
      [hdr("Dataset"), hdr("Field-miss risk\n(tok. → CRC)"), hdr("Span CovGap pp\n(tok. → CRC)"), hdr("Set size")],
      [cell("SROIE", { bold: true }), cell("1.000 → 0.091", { bold: true, color: HEX.teal }), cell("90.0 → 2.1", { bold: true, color: HEX.teal }), cell("+10%")],
      [cell("SROIE-rep."), cell("0.749 → 0.110"), cell("64.2 → 2.9"), cell("+11%")],
      [cell("WildReceipt"), cell("0.093 → 0.100"), cell("13.5 → 8.8"), cell("+1%")],
      [cell("FUNSD"), cell("0.054 → 0.058"), cell("6.4 → 5.0"), cell("+23%")],
      [cell("CORD"), cell("0.115 → 0.108"), cell("42.3 → 33.3"), cell("+224%", { bold: true, color: HEX.red })],
    ];
    s.addTable(rows, { x: 7.0, y: 1.45, w: 5.73, colW: [1.35, 1.6, 1.68, 1.1], rowH: [0.62, 0.48, 0.48, 0.48, 0.48, 0.48],
      border: { type: "solid", pt: 0.75, color: HEX.line }, margin: [0.03, 0.08, 0.03, 0.08], fontFace: "Calibri", objectName: nm("table_results") });
    T(s, [
      { text: "Risk", options: { bold: true } }, { text: " = realized per-document fraction of gold fields missed; the theorem bounds its expectation at α = 0.10. ", options: {} },
      { text: "Coverage, CovGap and set size are empirical.", options: { bold: true } },
      { text: " SpanCRC full-span coverage ± s.d. over calibration splits: SROIE 0.909 ± 0.011, SROIE-rep. 0.894 ± 0.020, WildReceipt 0.913 ± 0.004, FUNSD 0.944 ± 0.005.", options: { color: COL.slate } },
    ], { x: 7.0, y: 4.6, w: 5.73, h: 1.5, fontSize: 13 });
    source(s, "Main paper Table 2 and Fig. 1(a), Sec. 4.5; set-size change: Supp. Fig. S1. α = 0.10, mixed rule (m0 = 25).");
    addNotes(s, 8);
  }

  // =====================================================================
  // 9 — CORD trade-off
  // =====================================================================
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
    title(s, "CORD: limited field support makes the repair expensive");
    const cards = [
      ["42.3 → 33.3", "span CovGap (pp): a modest gain", COL.ink],
      ["0.90 → 2.92", "average set size (3.2×); 100% of tokens carry multi-label sets", COL.red],
      ["0.108", "realized risk on its single calibration split: above α, consistent with a bound on the expectation", COL.amber],
    ];
    cards.forEach(([big, small, col], i) => {
      const y = 1.45 + i * 1.5;
      box(s, MX, y, 5.5, 1.3, { fill: COL.mist, r: 0.08, name: "statcard" });
      T(s, big, { x: MX + 0.25, y: y + 0.12, w: 2.6, h: 1.05, fontSize: 32, bold: true, color: col, valign: "middle", fontFace: THEME.headFontFace });
      T(s, small, { x: MX + 2.85, y: y + 0.1, w: 2.5, h: 1.1, fontSize: 14, valign: "middle" });
    });
    T(s, [
      { text: "Why: ", options: { bold: true } },
      { text: "most field types fall back to the pooled threshold; thinly supported types get large thresholds (MENU.SUB.CNT: λ = 1.0), admitting their labels into most sets." },
    ], { x: MX, y: 5.95, w: 5.5, h: 0.65, fontSize: 14 });
    // right: guaranteed alternative
    const rx = 6.6;
    box(s, rx, 1.45, 6.13, 5.1, { fill: COL.white, line: COL.silver, lw: 1, r: 0.08, name: "altpanel" });
    T(s, "A repair that keeps a guarantee: level-calibrated interpolation", { x: rx + 0.25, y: 1.55, w: 5.7, h: 0.7, fontSize: 17, bold: true, color: COL.teal });
    eq(s, "eq_interp", rx + 0.25, 2.3, 19, "thr_t(mu) = min{1, tau lambda_t^A + (1 - tau) mu}");
    T(s, "Shape λ{_t}{^A} from calibration half A; one level μ chosen by CRC on half B (Supp. Prop. 3).", { x: rx + 0.25, y: 2.8, w: 5.7, h: 0.55, fontSize: 13, color: COL.slate });
    const hdr = (t) => ({ text: t, options: { bold: true, color: HEX.slate, fontSize: 13, fill: { color: HEX.mist } } });
    const c = (t, o = {}) => ({ text: t, options: Object.assign({ fontSize: 15, color: HEX.ink }, o) });
    s.addTable([
      [hdr("CORD, α = 0.10"), hdr("Deployed m0 rule"), hdr("Level-cal. τ = 0.5")],
      [c("Realized risk"), c("0.108"), c("0.035", { bold: true, color: HEX.teal })],
      [c("Span CovGap (pp)"), c("33.3"), c("18.4", { bold: true, color: HEX.teal })],
      [c("Avg. set size"), c("2.92"), c("1.12", { bold: true, color: HEX.teal })],
      [c("Empty sets"), c("0.0%"), c("0.0%")],
    ], { x: rx + 0.25, y: 3.4, w: 5.65, colW: [1.95, 1.75, 1.95], rowH: 0.38, border: { type: "solid", pt: 0.75, color: HEX.line },
      margin: [0.03, 0.08, 0.03, 0.08], fontFace: "Calibri", objectName: nm("table_interp") });
    box(s, rx + 0.25, 5.42, 5.65, 1.0, { fill: COL.amber, ft: 90, r: 0.06, name: "caveat" });
    T(s, [
      { text: "Pooled document-level bound only, not field-conditional. ", options: { bold: true } },
      { text: "Over-covers (μ ≈ 0.92). Needs enough calibration documents: on FUNSD (17 level documents) sets inflate to 4.69." },
    ], { x: rx + 0.4, y: 5.42, w: 5.4, h: 1.0, fontSize: 13, valign: "middle" });
    source(s, "Main paper Sec. 4.5, 5.2, 6; Supp. S4.3 (Prop. 3, Table S12), S8.1, Table S14");
    addNotes(s, 9);
  }

  // =====================================================================
  // 10 — Boundary-escape test
  // =====================================================================
  pres.addSection({ title: "Audit" });
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Audit" });
    title(s, "Boundary-escape audit: where inside a field do escapes land?");
    // examples (conceptual)
    const ex = [
      { lab: "Edge-concentrated", pat: [1, 0, 0, 0, 0, 1], note: "2 of 2 escapes on edges", col: COL.red },
      { lab: "Interior", pat: [0, 0, 1, 0, 0], note: "0 of 1 escape on edges", col: COL.ink },
      { lab: "Uninformative", pat: [0, 0, 0, 0], note: "no escape (k = 0): carries no position", col: COL.slate },
      { lab: "Blind spot", pat: [1, 0], note: "L ≤ 2: every position is an edge", col: COL.slate },
    ];
    T(s, "Conceptual spans  (red = escaped word: gold-tag score beyond τ = pooled λ)", { x: MX, y: 1.35, w: 6.2, h: 0.32, fontSize: 14, bold: true, color: COL.slate });
    ex.forEach((e, i) => {
      const y = 1.85 + i * 0.68;
      T(s, e.lab, { x: MX, y, w: 1.85, h: 0.42, fontSize: 15, bold: true, color: e.col, valign: "middle" });
      e.pat.forEach((p, j) => {
        const edge = j === 0 || j === e.pat.length - 1;
        chip(s, 2.5 + j * 0.42, y + 0.04, 0.34, 0.34, p ? "bad" : "neutral");
        if (edge && e.pat.length > 2) rule(s, 2.5 + j * 0.42, y + 0.43, 2.84 + j * 0.42, y + 0.43, COL.amber, 2.5);
      });
      T(s, e.note, { x: 5.15, y, w: 2.0, h: 0.42, fontSize: 12, color: COL.slate, valign: "middle" });
    });
    T(s, "amber underline = edge positions B{_s}", { x: 2.5, y: 4.55, w: 3.5, h: 0.26, fontSize: 11, italic: true, color: COL.amber });
    // statistic + null
    T(s, "Statistic over informative spans (0 < k{_s} < L{_s}) of field type t", { x: MX, y: 4.95, w: 6.5, h: 0.3, fontSize: 14, bold: true, color: COL.ink });
    eq(s, "eq_escape", MX, 5.3, 16, "E(s) = { i in [a,b) : 1 - p(g_i|x_i) > tau },  W_t = sum over informative spans of |E(s) ∩ B_s|");
    T(s, "Null: conditional on each span's (L{_s}, k{_s}), escaped positions are uniform within the span, independently across spans; Monte-Carlo p-value with B = 5,000 draws.",
      { x: MX, y: 5.82, w: 6.5, h: 0.75, fontSize: 13.5, color: COL.ink });
    // right: decision and caveats
    const rx = 7.45;
    box(s, rx, 1.35, 5.28, 2.45, { fill: COL.teal, ft: 90, r: 0.08, name: "decision" });
    T(s, "Decision rule", { x: rx + 0.22, y: 1.45, w: 4.8, h: 0.35, fontSize: 16, bold: true, color: COL.teal });
    T(s, [
      { text: "Holm across field types, FWER 0.10 (BH, BY reported)", options: { bullet: true, breakLine: true } },
      { text: "Document-blocked bootstrap interval for enrichment W{_t} / E[W{_t}]", options: { bullet: true, breakLine: true } },
      { text: "Claim a finding only when both agree", options: { bullet: true, bold: true } },
    ], { x: rx + 0.22, y: 1.85, w: 4.9, h: 1.85, fontSize: 14.5, paraSpaceAfter: 6 });
    box(s, rx, 3.95, 5.28, 2.6, { fill: COL.amber, ft: 90, r: 0.08, name: "caveats" });
    T(s, "What it does not establish", { x: rx + 0.22, y: 4.05, w: 4.8, h: 0.35, fontSize: 16, bold: true, color: COL.amber });
    T(s, [
      { text: "Cross-span independence is assumed; Holm cannot repair a raw p-value", options: { bullet: true, breakLine: true } },
      { text: "The bootstrap corroborates, but does not restore validity", options: { bullet: true, breakLine: true } },
      { text: "Spans of ≤ 2 words are untestable; ineligible fields are untested, not clean", options: { bullet: true, breakLine: true } },
      { text: "Not a general label-error detector", options: { bullet: true } },
    ], { x: rx + 0.22, y: 4.45, w: 4.9, h: 2.05, fontSize: 14, paraSpaceAfter: 4 });
    source(s, "Main paper Sec. 3.4, Eq. 4; Supp. S5 (Props. 1–2)");
    addNotes(s, 10);
  }

  // =====================================================================
  // 11 — Findings + repair
  // =====================================================================
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Audit" });
    title(s, "Two SROIE fields flagged; an independent repair silences both");
    // left: enrichment chart (Fig. 1b), top-to-bottom order as listed
    const f = [ // label, enrichment, rejected
      ["SROIE · ADDRESS", 4.07, true], ["CORD · SUB_TOTAL.SERVICE_PRICE", 2.00, false], ["SROIE · COMPANY", 1.80, true],
      ["FUNSD · QUESTION", 1.28, false], ["SROIE (repaired) · COMPANY", 1.15, false], ["WildReceipt · Prod_item_value", 1.00, false],
    ].reverse();
    T(s, "Edge enrichment  W{_t} / E[W{_t}]  (1.0 = no enrichment)", { x: MX, y: 1.35, w: 6.3, h: 0.32, fontSize: 15, bold: true });
    s.addChart(pres.charts.BAR, [
      { name: "Rejected (Holm, FWER 0.10)", labels: f.map((r) => r[0]), values: f.map((r) => (r[2] ? r[1] : 0)) },
      { name: "Not rejected", labels: f.map((r) => r[0]), values: f.map((r) => (r[2] ? 0 : r[1])) },
    ], Object.assign({}, AX, {
      x: MX - 0.1, y: 1.7, w: 6.5, h: 3.65, barDir: "bar", barGrouping: "stacked", barGapWidthPct: 45,
      chartColors: [HEX.red, HEX.silver], showLegend: true, legendPos: "b", catAxisLabelFontSize: 13,
      valAxisMinVal: 0, valAxisMaxVal: 4.5, valAxisMajorUnit: 1, valAxisLabelFormatCode: "0.0",
      showValue: true, dataLabelPosition: "inEnd", dataLabelFormatCode: "0.00;;;", dataLabelFontSize: 12, dataLabelColor: HEX.ink,
      objectName: nm("chart_enrichment"),
    }));
    T(s, [
      { text: "ADDRESS, COMPANY: Holm-adjusted p = 0.0008 (Monte-Carlo floor); Holm, BH, BY agree on all 3 splits. ", options: {} },
      { text: "0 of 16 eligible clean-benchmark field types rejected. ", options: { bold: true } },
      { text: "CORD SERVICE_PRICE: 2 informative spans → wide null, p = 0.25.", options: { color: COL.slate } },
    ], { x: MX, y: 5.45, w: 6.3, h: 1.15, fontSize: 13.5 });
    // right: before/after dumbbell with CIs (shapes on a 0–5 axis)
    const rx = 7.3, rw = 5.43, ax0 = rx + 1.45, aw = rw - 1.65, sx = (v) => ax0 + v / 5 * aw;
    box(s, rx, 1.35, rw, 5.25, { fill: COL.mist, r: 0.08, name: "repairpanel" });
    T(s, "Falsification: re-derive SROIE labels from the official Task-3 field strings (separate lineage), re-run the audit",
      { x: rx + 0.2, y: 1.45, w: rw - 0.4, h: 0.75, fontSize: 14, bold: true, color: COL.ink });
    const items = [
      ["ADDRESS", [4.07, 3.37, 4.91, "Holm p = 0.0008"], [0.79, 0.41, 1.10, "p = 0.8666"]],
      ["COMPANY", [1.80, 1.66, 1.94, "Holm p = 0.0008"], [1.15, 0.93, 1.38, "p = 0.2204"]],
    ];
    // legend + null line at 1.0
    s.addShape(pres.shapes.OVAL, { x: rx + 0.25, y: 2.3, w: 0.16, h: 0.16, fill: { color: COL.red }, line: { type: "none" }, objectName: nm("dot") });
    T(s, "original labels", { x: rx + 0.48, y: 2.24, w: 1.6, h: 0.28, fontSize: 12, bold: true, color: COL.red });
    s.addShape(pres.shapes.OVAL, { x: rx + 2.15, y: 2.3, w: 0.16, h: 0.16, fill: { color: COL.teal }, line: { type: "none" }, objectName: nm("dot") });
    T(s, "repaired labels", { x: rx + 2.38, y: 2.24, w: 1.8, h: 0.28, fontSize: 12, bold: true, color: COL.teal });
    const yTop = 2.6, yBot = 5.25;
    rule(s, sx(1), yTop, sx(1), yBot, COL.slate, 1.25, { dash: "dash" });
    items.forEach(([n, a, b], i) => {
      const yb = 2.75 + i * 1.3;
      T(s, n, { x: rx + 0.2, y: yb, w: 1.25, h: 0.9, fontSize: 15, bold: true, valign: "middle" });
      [[a, COL.red, "original", 0], [b, COL.teal, "repaired", 0.5]].forEach(([v, col, lab, off]) => {
        const y = yb + 0.2 + off;
        rule(s, sx(v[1]), y, sx(v[2]), y, col, 2.5);
        s.addShape(pres.shapes.OVAL, { x: sx(v[0]) - 0.09, y: y - 0.09, w: 0.18, h: 0.18, fill: { color: col }, line: { color: COL.white, width: 1 }, objectName: nm("dot") });
        const left = v[0] > 2.5, labX = left ? sx(v[1]) - 1.78 : sx(v[2]) + 0.1;
        T(s, `${v[0].toFixed(2)}  (${v[3]})`, { x: labX, y: y - 0.15, w: 1.7, h: 0.3, fontSize: 12, color: col, bold: true, align: left ? "right" : "left" });
      });
    });
    rule(s, ax0, yBot, ax0 + aw, yBot, COL.slate, 1);
    [0, 1, 2, 3, 4, 5].forEach((v) => { rule(s, sx(v), yBot, sx(v), yBot + 0.07, COL.slate, 1); T(s, String(v), { x: sx(v) - 0.2, y: yBot + 0.08, w: 0.4, h: 0.22, fontSize: 11, color: COL.slate, align: "center" }); });
    T(s, "dot = enrichment; bar = document-blocked 90% bootstrap interval", { x: rx + 0.2, y: yBot + 0.3, w: rw - 0.4, h: 0.25, fontSize: 11, color: COL.slate, align: "center" });
    T(s, "After repair: nothing rejected under any procedure on any split. The repair is the re-alignment, not the calibration.",
      { x: rx + 0.2, y: 5.85, w: rw - 0.4, h: 0.68, fontSize: 13.5, bold: true, color: COL.teal, valign: "middle" });
    source(s, "Main paper Sec. 4.6, Fig. 1(b), Sec. 4.9; Supp. Tables S17–S18, S6. First calibration split, B = 5,000.");
    addNotes(s, 11);
  }

  // =====================================================================
  // 12 — Closest prior work
  // =====================================================================
  pres.addSection({ title: "Context" });
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Context" });
    title(s, "Closest prior work certifies a different object");
    const hdr = (t, col = HEX.ink) => ({ text: t, options: { bold: true, color: col, fontSize: 15, fill: { color: HEX.mist }, valign: "middle" } });
    const rl = (t) => ({ text: t, options: { bold: true, color: HEX.slate, fontSize: 14, valign: "middle" } });
    const c = (t, o = {}) => ({ text: t, options: Object.assign({ fontSize: 14, color: HEX.ink, valign: "middle" }, o) });
    const k = { bold: true, color: HEX.teal };
    s.addTable([
      [hdr(""), hdr("Rombach & Mehdiyev (IJDAR 2026)"), hdr("Mondrian / clustered CP"), hdr("CaliDoc", HEX.teal)],
      [rl("Certified unit"), c("entity instance, scored at its first token"), c("token, per class or class cluster"), c("complete field span (max-token score)", k)],
      [rl("Exchangeable unit"), c("entity instances"), c("tokens"), c("documents", k)],
      [rl("Guaranteed quantity"), c("marginal coverage over entity instances"), c("class-conditional token coverage"), c("expected per-document field-miss risk", k)],
      [rl("Per-field calibration"), c("noted as future work"), c("per class; data-hungry"), c("per supported field type, pooled fallback", k)],
      [rl("Annotation audit"), c("—"), c("—"), c("boundary-escape test with explicit null", k)],
    ], { x: MX, y: 1.4, w: W - 2 * MX, colW: [2.2, 3.35, 3.0, 3.58], rowH: [0.55, 0.62, 0.48, 0.62, 0.55, 0.55],
      border: { type: "solid", pt: 0.75, color: HEX.line }, margin: [0.04, 0.1, 0.04, 0.1], fontFace: "Calibri", objectName: nm("table_related") });
    box(s, MX, 5.05, 6.0, 1.45, { fill: COL.mist, r: 0.08, name: "oppoint" });
    T(s, [
      { text: "Their operating point (WildReceipt, α = 0.02)", options: { bold: true, breakLine: true } },
      { text: "published 98.3% coverage, set size 1.38, 70% singletons; same construction on our probabilities 0.980 ± 0.002, 1.16, 88.8%. ", options: {} },
      { text: "Stated, not ranked: backbone, score, schema and split differ.", options: { bold: true, color: COL.red } },
    ], { x: MX + 0.2, y: 5.1, w: 5.65, h: 1.35, fontSize: 13.5, valign: "middle" });
    box(s, 6.85, 5.05, 5.88, 1.45, { fill: COL.teal, ft: 90, r: 0.08, name: "proxygap" });
    T(s, [
      { text: "Proxy vs field, SROIE at α = 0.10", options: { bold: true, breakLine: true } },
      { text: "first-token unit 0.897 vs full-span 0.856; token-marginal 0.000. The proxy-to-field gap depends on the dataset and operating point, so the delivered object should carry the guarantee." },
    ], { x: 7.05, y: 5.1, w: 5.55, h: 1.35, fontSize: 13.5, valign: "middle" });
    source(s, "Main paper Sec. 2.1–2.2, Sec. 4.4.1; Supp. Table S9");
    addNotes(s, 12);
  }

  // =====================================================================
  // 13 — Limitations
  // =====================================================================
  {
    const s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Context" });
    title(s, "Limitations, stated up front");
    const L = [
      [ic.risk, "A risk, not a coverage", "Bounds the expected missed-field fraction per document; one split can exceed α (CORD 0.108)."],
      [ic.layers, "Support-limited", "Below m0 = 25 calibration spans → pooled fallback. CORD: 42.3 → 33.3 pp at 3.2× set size; the interpolation fix keeps only the pooled bound."],
      [ic.ruler, "Short-span blind spot", "Spans of ≤ 2 words carry no positional information; WildReceipt is largely untestable."],
      [ic.link, "Null assumption", "Cross-span independence is assumed; the document-blocked bootstrap corroborates but does not restore validity."],
      [ic.target, "One real positive", "Real-data defect validation rests on SROIE; ADDRESS and COMPANY share the same receipts."],
      [ic.globe, "Scope", "Four English receipt/form corpora; LayoutLMv3 (Large, Base). Multilingual, long-document, OCR-free: untested."],
    ];
    L.forEach(([im, h, b], i) => {
      const col = i % 3, row = Math.floor(i / 3);
      const x = MX + col * 4.1, y = 1.45 + row * 2.6, w = 3.85, hh = 2.35;
      box(s, x, y, w, hh, { fill: COL.mist, r: 0.08, name: "limit" });
      s.addShape(pres.shapes.OVAL, { x: x + 0.22, y: y + 0.22, w: 0.62, h: 0.62, fill: { color: COL.white }, line: { color: COL.teal, width: 1 }, objectName: nm("iconbg") });
      s.addImage({ data: im, x: x + 0.35, y: y + 0.35, w: 0.36, h: 0.36, altText: h, objectName: nm("icon") });
      T(s, h, { x: x + 1.0, y: y + 0.22, w: w - 1.15, h: 0.62, fontSize: 18, bold: true, valign: "middle" });
      T(s, b, { x: x + 0.22, y: y + 0.98, w: w - 0.4, h: 1.3, fontSize: 14.5, color: COL.ink });
    });
    source(s, "Main paper Sec. 6 (Limitations), Sec. 3.4, 4.6; Supp. S2.1");
    addNotes(s, 13);
  }

  // =====================================================================
  // 14 — Takeaways + resources
  // =====================================================================
  {
    const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Context" });
    T(s, "Calibrate the object you deliver", { x: MX, y: 0.5, w: 12, h: 0.8, fontFace: THEME.headFontFace, fontSize: 36, bold: true, color: COL.white });
    const tk = [
      ["1", "Token-level reliability is not complete-field reliability", "SROIE: 89.8% of tokens covered, 0 of 420 fields."],
      ["2", "Make the delivered field the unit", "Document-level CRC bounds expected field-miss risk, with no retraining."],
      ["3", "Use the calibration to audit annotations", "A positional test with an explicit null localized a real defect; an independent repair silenced it."],
    ];
    tk.forEach(([n, h, b], i) => {
      const y = 1.55 + i * 1.22;
      s.addShape(pres.shapes.OVAL, { x: MX, y: y + 0.05, w: 0.62, h: 0.62, fill: { color: COL.teal }, line: { type: "none" }, objectName: nm("num") });
      T(s, n, { x: MX, y: y + 0.05, w: 0.62, h: 0.62, fontSize: 22, bold: true, color: COL.white, align: "center", valign: "middle" });
      T(s, h, { x: MX + 0.85, y, w: 11.2, h: 0.45, fontSize: 22, bold: true, color: COL.white });
      T(s, b, { x: MX + 0.85, y: y + 0.47, w: 11.2, h: 0.4, fontSize: 17, color: "CBD5E1" });
    });
    // resources
    const ry = 5.3;
    const res = [
      [ic.gh, "Code", "github.com/MHHamdan/CaliDoc"],
      [ic.book, "Supplement (22 pp.)", "release wacv2027-camera-ready"],
      [ic.play, "Colab demo", "tag wacv2027-demo-v1"],
      [ic.box, "Checkpoints", "Hugging Face collection"],
    ];
    res.forEach(([im, h, b], i) => {
      const x = MX + i * 3.08;
      s.addImage({ data: im, x, y: ry + 0.05, w: 0.38, h: 0.38, altText: h, objectName: nm("icon") });
      T(s, h, { x: x + 0.5, y: ry, w: 2.5, h: 0.3, fontSize: 14, bold: true, color: COL.white });
      T(s, b, { x: x + 0.5, y: ry + 0.3, w: 2.55, h: 0.3, fontSize: 12, color: "9FD3CC",
        hyperlink: { url: ["https://github.com/MHHamdan/CaliDoc", "https://github.com/MHHamdan/CaliDoc/releases/tag/wacv2027-camera-ready",
          "https://colab.research.google.com/github/MHHamdan/CaliDoc/blob/wacv2027-demo-v1/notebooks/CaliDoc_WACV2027_Demo.ipynb",
          "https://huggingface.co/collections/MHamdan/calidoc-wacv-2027-6ad41a62c6a81846dfeb44b4"][i] } });
    });
    T(s, "Thank you — questions welcome", { x: MX, y: 6.25, w: 7, h: 0.5, fontSize: 22, bold: true, color: "9FD3CC" });
    T(s, "Supported by NSERC Discovery Grant RGPIN-2025-05449", { x: 7.5, y: 6.35, w: 5.23, h: 0.35, fontSize: 13, color: "CBD5E1", align: "right" });
    addNotes(s, 14);
  }

  // =====================================================================
  // Backup
  // =====================================================================
  pres.addSection({ title: "Backup" });
  {
    const s = pres.addSlide({ masterName: "TITLE_DARK", sectionTitle: "Backup" });
    T(s, "Backup slides", { x: MX, y: 0.9, w: 12, h: 0.9, fontFace: THEME.headFontFace, fontSize: 40, bold: true, color: COL.white });
    const idx = [["B1", "Mondrian on SROIE"], ["B2", "Why the max score"], ["B3", "Small-calibration controls"], ["B4", "Audit size and power"], ["B5", "Label-error detectors"],
      ["B6", "Generality: Base backbone, clean calibration"], ["B7", "SROIE re-alignment lineage"], ["B8", "Reading α in deployment"], ["B9", "CaliDoc in action (CORD)"], ["B10", "Calibration and robustness"]];
    idx.forEach(([k, t], i) => {
      const x = MX + Math.floor(i / 5) * 6.1, y = 2.2 + (i % 5) * 0.85;
      T(s, k, { x, y, w: 0.85, h: 0.5, fontSize: 20, bold: true, color: "9FD3CC", valign: "middle" });
      T(s, t, { x: x + 0.9, y, w: 5.0, h: 0.5, fontSize: 20, color: COL.white, valign: "middle" });
    });
    addNotes(s, 15);
  }
  // B1 — Mondrian
  {
    const s = pres.addSlide({ masterName: "BACKUP", sectionTitle: "Backup" });
    title(s, "B1 · Mondrian fixes per-class coverage, still certifies tokens");
    const cls = ["O", "I-ADDRESS", "I-COMPANY", "B-ADDRESS", "B-COMPANY", "B-DATE", "B-TOTAL", "I-DATE", "I-TOTAL"];
    const marg = [96.1, 95.9, 2.5, 0, 0, 0, 0, 0, 0], mond = [89.3, 90.1, 89.9, 87.0, 89.5, 91.4, 88.3, 94.1, 100.0];
    s.addChart([
      { type: pres.charts.BAR, data: [{ name: "Single marginal threshold", labels: cls, values: marg }, { name: "Mondrian (class-conditional)", labels: cls, values: mond }],
        options: { barGrouping: "clustered", chartColors: [HEX.red, HEX.teal] } },
      { type: pres.charts.LINE, data: [{ name: "Target 90%", labels: cls, values: cls.map(() => 90) }], options: { chartColors: [HEX.ink], lineSize: 1.5, lineDataSymbol: "none", lineDash: "dash", showValue: false } },
    ], Object.assign({}, AX, { x: MX - 0.1, y: 1.35, w: 8.2, h: 5.1, barDir: "col", barGapWidthPct: 40, catAxisLabelFontSize: 12, showLegend: true, legendPos: "t",
      valAxisMinVal: 0, valAxisMaxVal: 100, valAxisMajorUnit: 25, showValue: false,
      valAxisTitle: "Per-class token coverage (%)", showValAxisTitle: true, valAxisTitleFontSize: 12, valAxisTitleColor: HEX.slate, valAxisTitleFontFace: "Calibri", objectName: nm("chart_mondrian") }));
    box(s, 8.5, 1.45, 4.23, 5.0, { fill: COL.mist, r: 0.08, name: "side" });
    T(s, [
      { text: "SROIE, LAC, 90% target", options: { bold: true, breakLine: true } },
      { text: "Mondrian: every class 87.0–100.0% at set size 0.90; CovGap 71.1 → 3.4 pp.", options: { bullet: true, breakLine: true } },
      { text: "Price: per-class quantiles span a factor of 367 (0.0027 for O → 0.9996 for I-TOTAL).", options: { bullet: true, breakLine: true } },
      { text: "The spread is a screening signal, not evidence: part of it is thin support (I-TOTAL n = 8).", options: { bullet: true, breakLine: true } },
      { text: "Still a token guarantee: the audit replaces the spread with a statistic that has a null.", options: { bullet: true, bold: true } },
    ], { x: 8.7, y: 1.6, w: 3.85, h: 4.7, fontSize: 15, paraSpaceAfter: 8 });
    source(s, "Main paper Table 1, Sec. 4.4.2");
    addNotes(s, 16);
  }
  // B2 — why max
  {
    const s = pres.addSlide({ masterName: "BACKUP", sectionTitle: "Backup" });
    title(s, "B2 · Other span scores certify other events");
    const sc = ["max (ours)", "mean", "2nd-largest", "within-span Q0.9"];
    s.addChart(pres.charts.BAR, [
      { name: "Rate of the score's own event", labels: sc, values: [0.886, 0.898, 0.886, 0.893] },
      { name: "True full-span coverage", labels: sc, values: [0.886, 0.762, 0.698, 0.840] },
    ], Object.assign({}, AX, { x: MX - 0.1, y: 1.35, w: 7.6, h: 5.1, barDir: "col", barGrouping: "clustered", barGapWidthPct: 50, chartColors: [HEX.silver, HEX.teal],
      showLegend: true, legendPos: "t", valAxisMinVal: 0.5, valAxisMaxVal: 1, valAxisMajorUnit: 0.1, valAxisLabelFormatCode: "0.0",
      showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", objectName: nm("chart_agg") }));
    box(s, 7.9, 1.45, 4.83, 5.0, { fill: COL.mist, r: 0.08, name: "side" });
    T(s, [
      { text: "SROIE, pooled CRC on each score", options: { bold: true, breakLine: true } },
      { text: "Each score lands near nominal on its own event, but only the max's event is complete-field delivery.", options: { bullet: true, breakLine: true } },
      { text: "Realized document field-miss risk: max 0.114, mean 0.238, 2nd-largest 0.302, Q0.9 0.160.", options: { bullet: true, breakLine: true } },
      { text: "Gap grows with span length: Spearman(length, score) 0.59 on SROIE vs 0.04 on WildReceipt (spans ≤ 2 words, all scores coincide).", options: { bullet: true, breakLine: true } },
      { text: "Diagnostic ablation, not proposed alternatives.", options: { bullet: true, bold: true } },
    ], { x: 8.1, y: 1.6, w: 4.45, h: 4.7, fontSize: 15, paraSpaceAfter: 8 });
    source(s, "Main paper Sec. 3.3, 4.5; Supp. S4.1–S4.2, Tables S10–S11 (first calibration split)");
    addNotes(s, 17);
  }
  // B3 — calibration size
  {
    const s = pres.addSlide({ masterName: "BACKUP", sectionTitle: "Backup" });
    title(s, "B3 · Small calibration counts do not explain a collapse to zero");
    const n = ["5", "10", "20", "50", "100"];
    T(s, "Matched-n control: fixed cohort of clean CORD classes, only n varies", { x: MX, y: 1.4, w: 6, h: 0.35, fontSize: 15, bold: true });
    s.addChart(pres.charts.LINE, [{ name: "Class-conditional coverage", labels: n, values: [0.810, 0.910, 0.954, 0.922, 0.905] }],
      Object.assign({}, AX, { x: MX - 0.1, y: 1.8, w: 6.1, h: 4.1, chartColors: [HEX.teal], lineSize: 3, lineDataSymbol: "circle", lineDataSymbolSize: 9,
        showLegend: false, valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.25, valAxisLabelFormatCode: "0.00",
        showValue: true, dataLabelPosition: "t", dataLabelFormatCode: "0.000", catAxisTitle: "calibration tokens per class (n)", showCatAxisTitle: true,
        catAxisTitleFontSize: 12, catAxisTitleColor: HEX.slate, catAxisTitleFontFace: "Calibri", objectName: nm("chart_matchedn") }));
    T(s, "Nine-point cost at n = 5, far from a collapse to zero", { x: MX, y: 5.95, w: 6, h: 0.35, fontSize: 14, italic: true, color: COL.teal, bold: true });
    const hdr = (t) => ({ text: t, options: { bold: true, color: HEX.slate, fontSize: 13, fill: { color: HEX.mist } } });
    const c = (t, o = {}) => ({ text: t, options: Object.assign({ fontSize: 15, color: HEX.ink }, o) });
    T(s, "Stronger baselines on corrected CORD (CovGap, pp)", { x: 7.05, y: 1.4, w: 5.7, h: 0.35, fontSize: 15, bold: true });
    s.addTable([
      [hdr("Method"), hdr("CovGap (pp)")],
      [c("Marginal"), c("45.8")], [c("Clustered CP"), c("45.9")], [c("Interpolation"), c("32.2")],
      [c("Full class-conditional"), c("16.4  (set 0.90 → 2.49)")],
    ], { x: 7.05, y: 1.85, w: 5.68, colW: [2.6, 3.08], rowH: 0.45, border: { type: "solid", pt: 0.75, color: HEX.line }, fontFace: "Calibri", margin: [0.03, 0.08, 0.03, 0.08], objectName: nm("table_condcp") });
    box(s, 7.05, 4.35, 5.68, 1.6, { fill: COL.mist, r: 0.08, name: "side" });
    T(s, "Feasibility bound: a class-conditional quantile exists only for n ≥ ⌈(1 − α)/α⌉ − 1 = 8 tokens; classes below it are marked, not scored. 24 CORD classes fall below it. Conditional CP at this granularity is a trade, not a free repair.",
      { x: 7.25, y: 4.4, w: 5.3, h: 1.5, fontSize: 14, valign: "middle" });
    source(s, "Main paper Sec. 4.7; Supp. Tables S28–S29");
    addNotes(s, 18);
  }
  // B4 — size & power
  {
    const s = pres.addSlide({ masterName: "BACKUP", sectionTitle: "Backup" });
    title(s, "B4 · Audit size and power (synthetic corruption)");
    const ds = ["FUNSD", "WildReceipt", "CORD"];
    T(s, "Size: familywise rejection rate under an exchangeable-position null (200 replicates, level 0.10)", { x: MX, y: 1.4, w: 6.1, h: 0.6, fontSize: 15, bold: true });
    s.addChart(pres.charts.BAR, [
      { name: "No correction", labels: ds, values: [0.140, 0.000, 0.045] },
      { name: "Holm", labels: ds, values: [0.025, 0.000, 0.000] },
    ], Object.assign({}, AX, { x: MX - 0.1, y: 2.05, w: 6.1, h: 3.9, barDir: "col", barGrouping: "clustered", chartColors: [HEX.silver, HEX.teal], showLegend: true, legendPos: "t",
      valAxisMinVal: 0, valAxisMaxVal: 0.16, valAxisMajorUnit: 0.04, valAxisLabelFormatCode: "0.00", showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.000", objectName: nm("chart_size") }));
    T(s, "Power: end-extension (one span extended onto a following O token)", { x: 7.05, y: 1.4, w: 5.7, h: 0.6, fontSize: 15, bold: true });
    const sp = ["1 span", "2 spans", "20 spans"];
    s.addChart(pres.charts.BAR, [{ name: "FUNSD QUESTION detection rate (30 seeds)", labels: sp, values: [0.53, 0.73, 0.93] }],
      Object.assign({}, AX, { x: 6.95, y: 2.05, w: 5.8, h: 2.6, barDir: "col", chartColors: [HEX.teal], showLegend: false, valAxisMinVal: 0, valAxisMaxVal: 1, valAxisMajorUnit: 0.25,
        valAxisLabelFormatCode: "0.00", showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", objectName: nm("chart_power") }));
    box(s, 7.05, 4.75, 5.68, 1.75, { fill: COL.amber, ft: 90, r: 0.08, name: "side" });
    T(s, [
      { text: "WildReceipt power 0.00 at every dose (≤ 2-word blind spot); start-side shift ≤ 0.20.", options: { bullet: true, breakLine: true } },
      { text: "Disclosure: FUNSD QUESTION clean raw p = 0.0362, Holm 0.1086 (not a finding).", options: { bullet: true, breakLine: true } },
      { text: "Interior mask added: Holm over 36 (field, mask) pairs still rejects only SROIE ADDRESS, COMPANY (edge mask).", options: { bullet: true } },
    ], { x: 7.2, y: 4.8, w: 5.45, h: 1.65, fontSize: 13.5, paraSpaceAfter: 4 });
    source(s, "Main paper Sec. 4.6; Supp. Tables S16, S21–S23");
    addNotes(s, 19);
  }
  // B5 — label-error detectors
  {
    const s = pres.addSlide({ masterName: "BACKUP", sectionTitle: "Backup" });
    title(s, "B5 · Label-error detectors: weaker ranker, different job");
    const hdr = (t) => ({ text: t, options: { bold: true, color: HEX.slate, fontSize: 14, fill: { color: HEX.mist } } });
    const c = (t, o = {}) => ({ text: t, options: Object.assign({ fontSize: 15, color: HEX.ink }, o) });
    T(s, "Planted CORD label noise, six conditions; corrupted tokens known by construction", { x: MX, y: 1.4, w: 12, h: 0.35, fontSize: 15, bold: true });
    s.addTable([
      [hdr("Detector"), hdr("Token-level AUPRC range"), hdr("Rank of corrupted class (of 54), class-targeted noise")],
      [c("Per-class coverage deficit (ours)", { bold: true }), c("0.007 – 0.169", { color: HEX.red, bold: true }), c("6th – 12th (never top-3); uniform 48th, BIO-shift 38th")],
      [c("Cleanlab token classification"), c("0.047 – 0.951"), c("21st – 46th (baselines)")],
      [c("AUM"), c("0.117 – 0.859"), c("—")],
      [c("Cartography, inductive-CP, max-softmax"), c("see Supp. Table S26"), c("—")],
    ], { x: MX, y: 1.85, w: W - 2 * MX, colW: [3.9, 3.0, 5.23], rowH: 0.5, border: { type: "solid", pt: 0.75, color: HEX.line }, fontFace: "Calibri", margin: [0.03, 0.1, 0.03, 0.1], objectName: nm("table_detectors") });
    box(s, MX, 4.6, W - 2 * MX, 1.7, { fill: COL.mist, r: 0.08, name: "side" });
    T(s, [
      { text: "Reading: ", options: { bold: true } },
      { text: "scoring a class-level statistic per token is the wrong use of it, and we report that directly. Localization comes from the boundary-escape test, which conditions on position and carries a null. CaliDoc is not claimed to outperform dedicated label-error detectors." },
    ], { x: MX + 0.25, y: 4.65, w: W - 2 * MX - 0.5, h: 1.6, fontSize: 16, valign: "middle" });
    source(s, "Main paper Sec. 4.8; Supp. S7, Table S26 (rank ranges for baselines as reported in main text)");
    addNotes(s, 20);
  }
  // B6 — generality
  {
    const s = pres.addSlide({ masterName: "BACKUP", sectionTitle: "Backup" });
    title(s, "B6 · Generality: second backbone, clean calibration");
    const hdr = (t) => ({ text: t, options: { bold: true, color: HEX.slate, fontSize: 13, fill: { color: HEX.mist }, valign: "middle" } });
    const c = (t, o = {}) => ({ text: t, options: Object.assign({ fontSize: 14, color: HEX.ink, valign: "middle" }, o) });
    s.addTable([
      [hdr("Setting (first calibration split)"), hdr("Full-span cov. tok. → field"), hdr("Doc field-miss risk tok. → field"), hdr("Avg. set size tok. → field")],
      [c("SROIE / Large"), c("0.000 → 0.909"), c("1.000 → 0.091"), c("0.90 → 0.99")],
      [c("SROIE / Base (3 seeds)", { bold: true }), c("0.000 → 0.901 ± 0.002", { bold: true }), c("1.000 → 0.099 ± 0.002", { bold: true }), c("0.90 → 0.99")],
      [c("FUNSD / Base (3 seeds)"), c("0.947 → 0.933"), c("0.055 → 0.071"), c("1.77 → 1.86")],
      [c("WildReceipt / Base (3 seeds)"), c("0.914 → 0.916"), c("0.101 → 0.099"), c("0.95 → 1.71", { color: HEX.red, bold: true })],
      [c("WildReceipt / Large, clean-cal (3 seeds)"), c("0.905 → 0.912"), c("0.110 → 0.101"), c("0.93 → 0.94")],
    ], { x: MX, y: 1.4, w: W - 2 * MX, colW: [3.9, 2.75, 2.85, 2.63], rowH: [0.6, 0.48, 0.48, 0.48, 0.48, 0.48], border: { type: "solid", pt: 0.75, color: HEX.line }, fontFace: "Calibri", margin: [0.03, 0.1, 0.03, 0.1], objectName: nm("table_general") });
    T(s, [
      { text: "Base: audit rejects exactly ADDRESS and COMPANY on all 3 seeds; no rejections on FUNSD, WildReceipt.", options: { bullet: true, breakLine: true } },
      { text: "Clean-cal: 100-document calibration split carved from the WildReceipt training partition, model retrained; official test used only for final evaluation. F1 94.05 ± 0.14 vs 94.38 ± 0.27; CovGap 13.8 → 10.8 pp; no rejections.", options: { bullet: true, breakLine: true } },
      { text: "Consistency across two LayoutLMv3 scales; broader architectural generality not claimed.", options: { bullet: true, bold: true } },
    ], { x: MX, y: 4.65, w: W - 2 * MX, h: 1.85, fontSize: 15, paraSpaceAfter: 6 });
    source(s, "Main paper Sec. 4.5; Supp. S2.1, Table S3 (± is s.d. over training seeds)");
    addNotes(s, 21);
  }
  // B7 — SROIE re-alignment
  {
    const s = pres.addSlide({ masterName: "BACKUP", sectionTitle: "Backup" });
    title(s, "B7 · SROIE: failed gate, repair as a separate lineage");
    const lab = ["Strict (gate)", "+ punctuation-insensitive", "+ numeric normalization"];
    s.addChart([
      { type: pres.charts.BAR, data: [{ name: "Field strings aligned to a contiguous OCR run (%)", labels: lab, values: [94.45, 97.25, 97.38] }], options: { chartColors: [HEX.teal] } },
      { type: pres.charts.LINE, data: [{ name: "Pre-registered gate 95%", labels: lab, values: [95, 95, 95] }], options: { chartColors: [HEX.red], lineSize: 2, lineDataSymbol: "none", lineDash: "dash", showValue: false } },
    ], Object.assign({}, AX, { x: MX - 0.1, y: 1.35, w: 6.3, h: 4.8, barDir: "col", barGapWidthPct: 60, showLegend: true, legendPos: "t", catAxisLabelFontSize: 12,
      valAxisMinVal: 90, valAxisMaxVal: 100, valAxisMajorUnit: 2, showValue: true, dataLabelPosition: "outEnd", dataLabelFormatCode: "0.00", objectName: nm("chart_align") }));
    box(s, 6.9, 1.45, 5.83, 5.0, { fill: COL.mist, r: 0.08, name: "side" });
    T(s, [
      { text: "Gate failed: 94.68% train (2,369/2,502), 94.02% test (1,305/1,388); pooled 94.45%, identical on three data lineages incl. the official site.", options: { bullet: true, breakLine: true } },
      { text: "Address fails most: 85.6% exact on test vs 95.1–99.1% for other fields.", options: { bullet: true, breakLine: true } },
      { text: "Residual: 0.10% non-contiguous, 2.52% no token subsequence (e.g. gold 81750 vs OCR B1750).", options: { bullet: true, breakLine: true } },
      { text: "Token level looks mixed: B-ADDRESS 0.000 → 0.444, I-ADDRESS 0.959 → 0.261; marginal CovGap 71.1 → 52.7 pp.", options: { bullet: true, breakLine: true } },
      { text: "Span and position level confirm: span coverage 0.257 → 0.894; ADDRESS enrichment 0.79, p = 0.8666.", options: { bullet: true, breakLine: true } },
      { text: "The repair does not retroactively pass the gate.", options: { bullet: true, bold: true } },
    ], { x: 7.1, y: 1.55, w: 5.45, h: 4.85, fontSize: 14, paraSpaceAfter: 6 });
    source(s, "Main paper Sec. 4.9; Supp. S6, Tables S24–S25");
    addNotes(s, 22);
  }
  // B8 — reading alpha
  {
    const s = pres.addSlide({ masterName: "BACKUP", sectionTitle: "Backup" });
    title(s, "B8 · Reading α and consuming the sets in a pipeline");
    const cards = [
      ["α = 0.10", "bounds the expected per-document fraction of missed gold fields: an average over documents, not a promise per receipt or per field type"],
      ["≤ 0.4", "expected missed fields per SROIE receipt (4 required fields); guaranteed for no receipt individually"],
      ["≈ α × fields × volume", "expected review burden; choosing α is a costing exercise against the cost of a miss"],
    ];
    cards.forEach(([b, t], i) => {
      const y = 1.45 + i * 1.6;
      box(s, MX, y, 6.0, 1.4, { fill: COL.teal, ft: 90, r: 0.08, name: "card" });
      T(s, b, { x: MX + 0.25, y: y + 0.1, w: 2.4, h: 1.2, fontSize: 24, bold: true, color: COL.teal, valign: "middle", fontFace: THEME.headFontFace });
      T(s, t, { x: MX + 2.7, y: y + 0.1, w: 3.15, h: 1.2, fontSize: 14, valign: "middle" });
    });
    const hdr = (t) => ({ text: t, options: { bold: true, color: HEX.slate, fontSize: 13, fill: { color: HEX.mist } } });
    const c = (t, o = {}) => ({ text: t, options: Object.assign({ fontSize: 15, color: HEX.ink }, o) });
    T(s, "Set shape under the deployed rule", { x: 6.95, y: 1.45, w: 5.7, h: 0.35, fontSize: 16, bold: true });
    s.addTable([
      [hdr("Dataset"), hdr("Token sets"), hdr("Practical cost")],
      [c("CORD"), c("100.0% multi-label (p90 size 3)"), c("triage every field")],
      [c("SROIE"), c("99.0% singleton"), c("near-singleton")],
      [c("WildReceipt"), c("92.9% singleton"), c("6.3% empty sets: abstain")],
    ], { x: 6.95, y: 1.9, w: 5.78, colW: [1.5, 2.45, 1.83], rowH: 0.5, border: { type: "solid", pt: 0.75, color: HEX.line }, fontFace: "Calibri", margin: [0.03, 0.08, 0.03, 0.08], objectName: nm("table_shape") });
    box(s, 6.95, 4.25, 5.78, 1.9, { fill: COL.amber, ft: 90, r: 0.08, name: "side" });
    T(s, "Per-field or adaptive α would need a guarantee we do not have. CaliDoc decodes no BIO-consistent alternatives: multi-label sets are consumed by ranking, format rules, or routing to review.",
      { x: 7.15, y: 4.3, w: 5.4, h: 1.8, fontSize: 14.5, valign: "middle" });
    source(s, "Main paper Sec. 5.2; Supp. S8.1");
    addNotes(s, 23);
  }
  // B9 — CaliDoc in action
  {
    const s = pres.addSlide({ masterName: "BACKUP", sectionTitle: "Backup" });
    title(s, "B9 · CaliDoc in action: routing a real CORD receipt");
    const im = path.join(ASSETS, "cord_test_0014_showcase.png");
    const { w, h } = imageSize(im); const iw = 10.9, ih = iw * h / w;
    s.addImage({ path: im, x: (W - iw) / 2, y: 1.3, w: iw, h: ih, altText: "CORD v2 test receipt cord_test_0014 with 25 extracted fields coloured by calibrated-set status (CC BY 4.0)", objectName: nm("showcase") });
    T(s, [
      { text: "25 fields extracted, 24 match gold; all 25 routed to review (CORD sets are multi-label). ", options: {} },
      { text: "The only error, Coupon 100,000 (predicted SUB_TOTAL.ETC, gold TOTAL.TOTAL_ETC), is the only conflict: ", options: {} },
      { text: "worst word 0.80 vs pooled threshold 0.189. ", options: { bold: true } },
      { text: "One illustrative case, not a sensitivity result; routing is operational, not a correctness probability.", options: { italic: true, color: COL.slate } },
    ], { x: MX, y: 1.3 + ih + 0.12, w: W - 2 * MX, h: 0.75, fontSize: 13.5 });
    source(s, "Supp. S12; CORD v2 (Park et al. 2019), CC BY 4.0; released checkpoint MHamdan/calidoc-layoutlmv3-large-cord, α = 0.10, m0 = 25");
    addNotes(s, 24);
  }
  // B10 — calibration & robustness
  {
    const s = pres.addSlide({ masterName: "BACKUP", sectionTitle: "Backup" });
    title(s, "B10 · Calibration, robustness, uncertainty baselines");
    const cards = [
      ["ECE after temperature scaling", "1.27 ± 0.22% CORD · 1.81 ± 0.08% WildReceipt · 6.37 ± 0.76% FUNSD (3 seeds)"],
      ["9 image perturbations × 3 severities", "mean F1 drop 0.56 pp (FUNSD); mean coverage drop < 0.15 pp everywhere (worst cell 1.53 pp). Empirical only: perturbation breaks exchangeability."],
      ["MC Dropout (CORD, seed 42)", "ECE 1.57 → 2.77%, coverage 87.9 → 78.6%, at 20× inference cost"],
      ["Empty LAC sets (CORD, 90%)", "9.9% of tokens; 68.2% of those would have been classified correctly"],
    ];
    cards.forEach(([h, b], i) => {
      const col = i % 2, row = Math.floor(i / 2);
      const x = MX + col * 6.13, y = 1.45 + row * 2.55;
      box(s, x, y, 5.95, 2.3, { fill: COL.mist, r: 0.08, name: "card" });
      T(s, h, { x: x + 0.25, y: y + 0.2, w: 5.45, h: 0.45, fontSize: 18, bold: true, color: COL.teal });
      T(s, b, { x: x + 0.25, y: y + 0.75, w: 5.45, h: 1.4, fontSize: 16 });
    });
    source(s, "Main paper Sec. 4.4, 4.10; Supp. S8–S9");
    addNotes(s, 25);
  }

  await pres.writeFile({ fileName: OUT });
  await fixPackage(OUT);
  if (SKILL) {
    const { applyTheme } = require(path.join(SKILL, "scripts", "apply_theme.js"));
    await applyTheme(OUT, THEME);
  }
  console.log("wrote", OUT);
})().catch((e) => { console.error(e); process.exit(1); });
