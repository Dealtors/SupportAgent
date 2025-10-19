const textarea = document.querySelector('.textarea');
const sendBtn  = document.querySelector('.send-btn');
const thread   = document.querySelector('.thread');

// автоскролл и ограничение высоты, чтобы не уезжало под композер
function ensureScrollable() {
  if (!thread.style.maxHeight) {
    thread.style.maxHeight = 'calc(100vh - 210px)';
    thread.style.overflowY = 'auto';
  }
}
function scrollBottom() {
  ensureScrollable();
  thread.scrollTop = thread.scrollHeight;
}

function escapeHTML(s='') {
  return s.replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function probClass(p) {
  if (p >= 0.8) return '';       // зелёная
  if (p >= 0.5) return ' mid';   // жёлтая
  return ' low';                 // розовая
}

function renderUser(text) {
  const html = `
  <article class="msg user">
    <div class="msg-avatar">🧑</div>
    <div class="msg-bubble">
      <div class="msg-card"><p>${escapeHTML(text)}</p></div>
    </div>
  </article>`;
  thread.insertAdjacentHTML('beforeend', html);
  scrollBottom();
}

function renderBot({ id, answer, confidence }) {
  const prob = typeof confidence === 'number' ? Math.round(confidence * 100) : null;
  const pill = prob != null
    ? `<button class="prob mono${probClass(confidence)}" data-toggle="meta" type="button">Вероятность: ${prob}%</button>`
    : '';
  const html = `
  <article class="msg bot" data-id="${id || ''}">
    <div class="msg-avatar">🤖</div>
    <div class="msg-bubble">
      <div class="msg-card">
        ${pill}
        <div class="msg-text"><p>${escapeHTML(answer || 'Ответ не получен')}</p></div>
        <div class="msg-meta" hidden>
          <div class="small">ID: ${id || '—'}</div>
        </div>
        <div class="msg-actions">
          <button class="icon-btn action-reply"  title="Ответить">💬</button>
          <button class="icon-btn action-email"  title="Отправить на почту">✉️</button>
          <button class="icon-btn action-copy"   title="Копировать">📋</button>
          <button class="icon-btn action-report" title="Скачать отчёт">📄</button>
          <button class="thumb-btn up action-like"   title="Полезно">👍</button>
          <button class="thumb-btn down action-dislike" title="Неполезно">👎</button>
        </div>
      </div>
    </div>
  </article>`;
  thread.insertAdjacentHTML('beforeend', html);
  wireActions(thread.lastElementChild);
  scrollBottom();
}

function mailto(subject, body) {
  const s = encodeURIComponent(subject || '');
  const b = encodeURIComponent(body || '');
  window.location.href = `mailto:?subject=${s}&body=${b}`;
}

function download(name, text) {
  const blob = new Blob([text], {type:'application/json'});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = name;
  document.body.appendChild(a); a.click();
  setTimeout(()=>{ URL.revokeObjectURL(url); a.remove(); }, 0);
}

async function sendFeedback(kind) {
  try {
    await fetch('/feedback', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ ticket_id: 'ui', feedback: kind })
    });
  } catch {}
}

function wireActions(root) {
  const text = root.querySelector('.msg-text')?.innerText || '';
  const id   = root.dataset.id || 'ui';

  root.querySelector('.action-reply')?.addEventListener('click', () => {
    textarea.focus();
  });

  root.querySelector('.action-email')?.addEventListener('click', () => {
    mailto('Ответ техподдержки', text);
  });

  root.querySelector('.action-copy')?.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(text);
      root.querySelector('.action-copy').textContent = '✅';
      setTimeout(()=> root.querySelector('.action-copy').textContent = '📋', 1200);
    } catch {}
  });

  root.querySelector('.action-report')?.addEventListener('click', () => {
    const payload = {
      id, created_at: new Date().toISOString(),
      answer: text
    };
    download(`report-${id}.json`, JSON.stringify(payload, null, 2));
  });

  root.querySelector('.action-like')?.addEventListener('click', () => {
    root.querySelector('.action-like').classList.add('active');
    root.querySelector('.action-dislike')?.classList.remove('active');
    sendFeedback('полезно');
  });

  root.querySelector('.action-dislike')?.addEventListener('click', () => {
    root.querySelector('.action-dislike').classList.add('active');
    root.querySelector('.action-like')?.classList.remove('active');
    sendFeedback('неполезно');
  });

  root.querySelector('[data-toggle="meta"]')?.addEventListener('click', () => {
    const meta = root.querySelector('.msg-meta');
    if (meta) meta.hidden = !meta.hidden;
  });
}

async function sendMessage() {
  const text = textarea.value.trim();
  if (!text) return;

  renderUser(text);
  textarea.value = ''; // очищаем поле ввода

  try {
    const res = await fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    // из бэка: answer, classification.confidence
    const confidence = data?.classification?.confidence ?? null;
    renderBot({ id: data?.message?.id || data?.ticket_id || null, answer: data.answer, confidence });
  } catch (e) {
    renderBot({ answer: 'Ошибка соединения с сервером.' });
  }
}

if (sendBtn && textarea) {
  sendBtn.addEventListener('click', sendMessage);
  textarea.addEventListener('keydown', e => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') sendMessage();
  });
}

// Чистый старт: если вдруг в HTML кто-то оставил демо-сообщения — удалим
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('.thread .msg').forEach(n => n.remove());
});
