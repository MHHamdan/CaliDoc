// Writes ../CaliDoc_WACV2027_Speaker_Notes.md from notes.js (the same source the PPTX notes use).
const fs = require("fs");
const path = require("path");
const NOTES = require("./notes.js");

const mmss = (s) => `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
const main = NOTES.filter((n) => !n.backup);
const total = main.reduce((a, n) => a + n.time, 0);
const words = main.reduce((a, n) => a + n.say.join(" ").split(/\s+/).length, 0);

const out = [];
out.push("# CaliDoc: WACV 2027 Talk, Speaker Notes");
out.push("");
out.push("Paper #1597, *CaliDoc: Calibrated Field-Level Risk and Annotation Auditing for Document Information Extraction*, Mohammed Hamdan and Mohamed Cheriet (Synchromedia Laboratory, ÉTS Montréal).");
out.push("");
out.push(`Main talk: ${main.length} slides, ${words} spoken words, about ${mmss(total)} at 145 words per minute. Backup slides B1–B10 are for Q&A only. These notes are generated from \`source/notes.js\`, the same text embedded in the PPTX Notes pages.`);
out.push("");
out.push("Conventions: `[pause]`, `[slow]` and `[close]` are delivery cues and are not spoken. Every number is traced in `CaliDoc_WACV2027_Claim_Audit.md`.");
out.push("");
out.push("## Timing at a glance");
out.push("");
out.push("| # | Slide | Time | Cumulative |");
out.push("|---|---|---|---|");
let cum = 0;
for (const n of main) { cum += n.time; out.push(`| ${n.id} | ${n.title} | ${mmss(n.time)} | ${mmss(cum)} |`); }
out.push("");
cum = 0;
for (const n of NOTES) {
  out.push(n.backup ? `## ${n.title} (backup)` : `## Slide ${n.id}: ${n.title}`);
  out.push("");
  if (!n.backup) { cum += n.time; out.push(`**Time:** ${mmss(n.time)} (cumulative ${mmss(cum)})`); out.push(""); }
  out.push("**Say:**");
  out.push("");
  for (const p of n.say) { out.push("> " + p); out.push(">"); }
  out.pop(); out.push("");
  if (n.emphasize) out.push(`- **Emphasize:** ${n.emphasize}`);
  if (n.why) out.push(`- **Why it matters:** ${n.why}`);
  if (n.transition) out.push(`- **Transition:** ${n.transition}`);
  if (n.avoid) out.push(`- **Avoid:** ${n.avoid}`);
  if (n.sources) out.push(`- **Sources:** ${n.sources}`);
  out.push("");
}
fs.writeFileSync(path.join(__dirname, "..", "CaliDoc_WACV2027_Speaker_Notes.md"), out.join("\n"));
console.log("wrote speaker notes:", main.length, "main slides,", words, "words,", mmss(total));
