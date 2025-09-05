async function loadCSV(path) {
  const res = await fetch(path, {cache: "no-store"});
  if (!res.ok) throw new Error(`Failed to fetch ${path}: ${res.status}`);
  const text = await res.text();
  return parseCSV(text);
}

function parseCSV(text) {
  const lines = text.trim().split(/\r?\n/);
  const header = lines.shift().split(",");
  return lines.map(line => {
    // naive split (no embedded commas expected in our simple summary)
    const cells = line.split(",");
    const obj = {};
    header.forEach((h, i) => obj[h] = cells[i]);
    return obj;
  });
}

function fmtPct(x) {
  const f = Number(x);
  if (Number.isFinite(f)) return (f * 100).toFixed(2) + "%";
  return x;
}

function addRow(obj) {
  const tr = document.createElement("tr");
  const td = (t) => { const el = document.createElement("td"); el.textContent = t; return el; };
  tr.appendChild(td(obj.scope));
  tr.appendChild(td(obj.lang));
  tr.appendChild(td(fmtPct(obj.wer)));
  tr.appendChild(td(fmtPct(obj.cer)));
  tr.appendChild(td(obj.num_utts));
  document.getElementById("tbody").appendChild(tr);
}

(async () => {
  try {
    const rows = await loadCSV("../results/results.csv");
    rows.forEach(addRow);
  } catch (e) {
    console.error(e);
    const tb = document.getElementById("tbody");
    const tr = document.createElement("tr");
    const td = document.createElement("td");
    td.colSpan = 5;
    td.textContent = "No results found. Generate results/results.csv and push to main.";
    tr.appendChild(td);
    tb.appendChild(tr);
  }
})();