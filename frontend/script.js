// ============================================================
// Contract Risk Analyzer — Frontend JavaScript
// Handles tab switching, API calls, and result rendering
// ============================================================

const API_BASE = window.location.origin;

// ── Sample contract for demo ──────────────────────────────────
const SAMPLE_CONTRACT = `SOFTWARE LICENSE AGREEMENT

This Software License Agreement ("Agreement") is entered into as of January 15, 2024,
between Acme Technology Corp. ("Licensor") and Beta Solutions LLC ("Licensee").

TERM
This Agreement commences on the Effective Date and continues for one (1) year.
This Agreement shall automatically renew for successive one-year periods unless
either party provides written notice of non-renewal at least 90 days prior to the
end of the then-current term.

LIMITATION OF LIABILITY
IN NO EVENT SHALL LICENSOR BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, OR
CONSEQUENTIAL DAMAGES. LICENSOR'S TOTAL CUMULATIVE LIABILITY SHALL NOT EXCEED THE
FEES PAID BY LICENSEE IN THE TWELVE MONTHS PRECEDING THE CLAIM.

CONFIDENTIALITY
Each party agrees to maintain the confidentiality of the other party's Confidential
Information and not to disclose such information to any third parties.

NON-COMPETE
During the term and for two (2) years thereafter, Licensee agrees not to develop,
market, or distribute any software product that competes with Licensor's products.

INTELLECTUAL PROPERTY
All work product, inventions, and deliverables created under this Agreement shall be
considered work for hire and shall be the exclusive property of Licensor. Licensee
hereby assigns all intellectual property rights to Licensor.

GOVERNING LAW
This Agreement shall be governed by the laws of the State of Delaware. Any disputes
shall be resolved through binding arbitration under AAA rules.

PAYMENT
Licensee shall pay the License Fee of $50,000 per year within 30 days of invoice.
Late payments shall accrue interest at 1.5% per month.`;


// ── Tab Switching ─────────────────────────────────────────────
function switchTab(tab) {
  // Hide both panels
  document.getElementById('tab-text').style.display = 'none';
  document.getElementById('tab-pdf').style.display = 'none';

  // Remove active from all buttons
  document.getElementById('tab-text-btn').classList.remove('active');
  document.getElementById('tab-pdf-btn').classList.remove('active');

  // Show the selected tab
  document.getElementById('tab-' + tab).style.display = 'block';
  document.getElementById('tab-' + tab + '-btn').classList.add('active');

  // Hide any errors
  hideError();
}


// ── Load Sample Contract ──────────────────────────────────────
function loadSample() {
  document.getElementById('contractText').value = SAMPLE_CONTRACT;
}


// ── File selection handler ────────────────────────────────────
function handleFileSelect() {
  const fileInput = document.getElementById('pdfFile');
  const nameDisplay = document.getElementById('file-name');
  if (fileInput.files.length > 0) {
    nameDisplay.textContent = '✅ ' + fileInput.files[0].name;
    nameDisplay.style.display = 'block';
  }
}


// ── Safe fetch helper — always returns JSON or throws a clean error ──
async function safeFetch(url, options) {
  // 90-second timeout (first run may load spaCy model)
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), 90000);

  try {
    const response = await fetch(url, { ...options, signal: controller.signal });
    clearTimeout(timer);

    // Parse response body — handle both JSON and plain-text errors
    let data;
    const contentType = response.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      data = await response.json();
    } else {
      const rawText = await response.text();
      // Try parsing as JSON anyway (FastAPI sometimes omits content-type)
      try { data = JSON.parse(rawText); }
      catch { throw new Error('Server error: ' + rawText.substring(0, 200)); }
    }

    if (!response.ok) {
      throw new Error(data.detail || data.message || 'Server returned status ' + response.status);
    }
    return data;

  } catch (err) {
    clearTimeout(timer);
    if (err.name === 'AbortError') {
      throw new Error('Request timed out (90s). The server may still be loading AI models — please try again in a moment.');
    }
    throw err;
  }
}


// ── Analyze Text ──────────────────────────────────────────────
async function analyzeText() {
  const text = document.getElementById('contractText').value.trim();

  if (!text || text.length < 50) {
    showError('Please enter at least 50 characters of contract text.');
    return;
  }

  setLoading('text', true);
  hideError();

  try {
    const data = await safeFetch(API_BASE + '/analyze-text/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text: text, title: 'Manual Analysis' })
    });
    renderResults(data);
  } catch (err) {
    showError(err.message);
    console.error('analyzeText error:', err);
  } finally {
    setLoading('text', false);
  }
}


// ── Upload PDF ────────────────────────────────────────────────
async function uploadPDF() {
  const fileInput = document.getElementById('pdfFile');

  if (!fileInput.files || fileInput.files.length === 0) {
    showError('Please select a PDF file first.');
    return;
  }

  setLoading('pdf', true);
  hideError();

  const formData = new FormData();
  formData.append('file', fileInput.files[0]);

  try {
    const data = await safeFetch(API_BASE + '/upload-contract/', {
      method: 'POST',
      body: formData
    });
    renderResults(data);
  } catch (err) {
    showError(err.message);
    console.error('uploadPDF error:', err);
  } finally {
    setLoading('pdf', false);
  }
}


// ── Render Results ────────────────────────────────────────────
function renderResults(data) {
  // Show the results section
  const section = document.getElementById('results-section');
  section.style.display = 'block';
  section.scrollIntoView({ behavior: 'smooth', block: 'start' });

  const risk = data.risk_analysis || {};
  const summary = data.summary || {};

  // ── Risk Banner ──
  const grade = summary.risk_grade || risk.risk_grade || '?';
  const score = summary.risk_score ?? risk.risk_score ?? 0;
  const label = summary.risk_label || risk.risk_label || '';
  const color = getRiskColor(grade);

  document.getElementById('res-grade').textContent = grade;
  document.getElementById('res-grade').style.color = color;
  document.getElementById('res-score').textContent = score + '/100';
  document.getElementById('res-label').textContent = label;
  document.getElementById('res-label').style.color = color;

  document.getElementById('res-counts').innerHTML =
    `<span class="risk-count-item">🔴 High: <strong>${risk.high_risk_clauses || 0}</strong></span>` +
    `<span class="risk-count-item">🟠 Medium: <strong>${risk.medium_risk_clauses || 0}</strong></span>` +
    `<span class="risk-count-item">🟢 Low: <strong>${risk.low_risk_clauses || 0}</strong></span>` +
    `<span class="risk-count-item">📄 Words: <strong>${data.word_count || 0}</strong></span>`;

  // Meta
  document.getElementById('results-meta').textContent =
    'Contract ID: ' + data.contract_id + ' | ' + data.filename + ' | Analyzed: ' + new Date(data.analyzed_at).toLocaleString();

  // ── Entities ──
  renderEntities(data.entities || {});

  // ── Dates ──
  renderDates(data.dates_found || []);

  // ── Structure ──
  renderStructure(data.structure_analysis || {});

  // ── Complexity ──
  renderComplexity(data.complexity_metrics || {});

  // ── Clauses ──
  renderClauses(risk.detected_clauses || []);

  // ── Recommendations ──
  renderRecommendations(risk.recommendations || []);
}


function renderEntities(entities) {
  const el = document.getElementById('res-entities');
  let html = '';

  if (entities.ORGANIZATIONS && entities.ORGANIZATIONS.length > 0) {
    html += '<div class="entity-group">';
    html += '<div class="entity-group-label">Organizations</div>';
    html += '<div class="entity-pills">';
    entities.ORGANIZATIONS.forEach(org => {
      html += `<span class="pill">${escapeHtml(org)}</span>`;
    });
    html += '</div></div>';
  }

  if (entities.DATES && entities.DATES.length > 0) {
    html += '<div class="entity-group" style="margin-top:10px;">';
    html += '<div class="entity-group-label">Dates</div>';
    html += '<div class="entity-pills">';
    entities.DATES.forEach(d => {
      html += `<span class="pill date">${escapeHtml(d)}</span>`;
    });
    html += '</div></div>';
  }

  if (entities.MONETARY_VALUES && entities.MONETARY_VALUES.length > 0) {
    html += '<div class="entity-group" style="margin-top:10px;">';
    html += '<div class="entity-group-label">Monetary Values</div>';
    html += '<div class="entity-pills">';
    entities.MONETARY_VALUES.forEach(m => {
      html += `<span class="pill money">${escapeHtml(m)}</span>`;
    });
    html += '</div></div>';
  }

  el.innerHTML = html || '<p class="empty-msg">No entities extracted. Make sure spaCy is installed.</p>';
}


function renderDates(dates) {
  const el = document.getElementById('res-dates');
  if (!dates || dates.length === 0) {
    el.innerHTML = '<p class="empty-msg">No dates found in the contract.</p>';
    return;
  }

  let html = '<div class="entity-pills">';
  dates.forEach(d => {
    html += `<span class="pill date">📅 ${escapeHtml(d)}</span>`;
  });
  html += '</div>';
  el.innerHTML = html;
}


function renderStructure(structure) {
  const el = document.getElementById('res-structure');
  const completeness = structure.completeness_score || 0;
  const grade = structure.completeness_grade || 'Unknown';
  const missing = structure.critical_clauses_missing || [];
  const present = structure.critical_clauses_present || [];

  let html = '';
  html += infoRow('Completeness Score', completeness + '%');
  html += infoRow('Status', grade);
  html += infoRow('Sections Found', structure.total_sections_found || 0);
  html += infoRow('Critical Clauses Present', present.length + ' of ' + (present.length + missing.length));

  if (missing.length > 0) {
    html += '<div style="margin-top:10px;">';
    html += '<div class="info-label" style="margin-bottom:6px;">Missing Clauses:</div>';
    missing.forEach(m => {
      html += `<span class="pill" style="background:#fef2f2;color:#b91c1c;border-color:#fca5a5;margin:2px;">⚠️ ${m}</span> `;
    });
    html += '</div>';
  }

  el.innerHTML = html;
}


function renderComplexity(complexity) {
  const el = document.getElementById('res-complexity');
  let html = '';
  html += infoRow('Complexity Level', complexity.complexity_level || '-');
  html += infoRow('Word Count', complexity.word_count || '-');
  html += infoRow('Sentence Count', complexity.sentence_count || '-');
  html += infoRow('Avg Sentence Length', (complexity.avg_sentence_length_words || '-') + ' words');
  html += infoRow('Legal Jargon Terms', complexity.legal_jargon_count || 0);

  if (complexity.complexity_note) {
    html += `<p style="font-size:12px;color:#64748b;margin-top:10px;">${complexity.complexity_note}</p>`;
  }

  el.innerHTML = html;
}


function renderClauses(clauses) {
  const el = document.getElementById('res-clauses');
  if (!clauses || clauses.length === 0) {
    el.innerHTML = '<p class="empty-msg">✅ No major risk clauses detected.</p>';
    return;
  }

  let html = '';
  clauses.forEach(clause => {
    html += `<div class="clause-item ${clause.risk_level}">`;
    html += '<div class="clause-header">';
    html += `<span class="risk-badge ${clause.risk_level}">${clause.risk_level}</span>`;
    html += `<span class="clause-name">${escapeHtml(clause.description)}</span>`;
    html += '</div>';
    if (clause.evidence_snippet) {
      html += `<div class="clause-evidence">"...${escapeHtml(clause.evidence_snippet.substring(0, 150))}..."</div>`;
    }
    if (clause.advice) {
      html += `<div class="clause-advice">💡 ${escapeHtml(clause.advice)}</div>`;
    }
    html += '</div>';
  });

  el.innerHTML = html;
}


function renderRecommendations(recs) {
  const el = document.getElementById('res-recommendations');
  if (!recs || recs.length === 0) {
    el.innerHTML = '<li>No specific recommendations.</li>';
    return;
  }

  let html = '';
  recs.forEach(rec => {
    html += `<li>${escapeHtml(rec)}</li>`;
  });
  el.innerHTML = html;
}


// ── Helper: info row ──────────────────────────────────────────
function infoRow(label, value) {
  return `<div class="info-row">
    <span class="info-label">${label}</span>
    <span class="info-value">${value}</span>
  </div>`;
}


// ── Helper: get color by risk grade ──────────────────────────
function getRiskColor(grade) {
  const colors = { 'A': '#16a34a', 'B': '#65a30d', 'C': '#d97706', 'D': '#ea580c', 'F': '#dc2626' };
  return colors[grade] || '#64748b';
}


// ── Helper: escape HTML ───────────────────────────────────────
function escapeHtml(text) {
  if (!text) return '';
  return String(text)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}


// ── Loading state ─────────────────────────────────────────────
function setLoading(tab, isLoading) {
  const loader = document.getElementById('loading-' + tab);
  if (loader) loader.style.display = isLoading ? 'block' : 'none';
}


// ── Error handling ────────────────────────────────────────────
function showError(msg) {
  const box = document.getElementById('error-box');
  const msgEl = document.getElementById('error-msg');
  msgEl.textContent = msg;
  box.style.display = 'block';
}

function hideError() {
  document.getElementById('error-box').style.display = 'none';
}


// ── Reset form ────────────────────────────────────────────────
function resetForm() {
  document.getElementById('contractText').value = '';
  document.getElementById('results-section').style.display = 'none';
  hideError();
}


// ── Smooth scroll for nav links ───────────────────────────────
document.querySelectorAll('a[href^="#"]').forEach(link => {
  link.addEventListener('click', function (e) {
    const targetId = this.getAttribute('href').substring(1);
    const target = document.getElementById(targetId);
    if (target) {
      e.preventDefault();
      target.scrollIntoView({ behavior: 'smooth' });
    }
  });
});
