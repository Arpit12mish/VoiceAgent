let lastPdfB64 = null;

function base64ToUint8Array(base64) {
  const binary_string = atob(base64);
  const len = binary_string.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) bytes[i] = binary_string.charCodeAt(i);
  return bytes;
}

async function renderPdf(pdfB64) {
  const pagesEl = document.getElementById("pages");
  pagesEl.innerHTML = "";

  pdfjsLib.GlobalWorkerOptions.workerSrc =
    "https://cdnjs.cloudflare.com/ajax/libs/pdf.js/4.2.67/pdf.worker.min.js";

  const pdfData = base64ToUint8Array(pdfB64);
  const loadingTask = pdfjsLib.getDocument({ data: pdfData });
  const pdf = await loadingTask.promise;

  for (let pageNum = 1; pageNum <= pdf.numPages; pageNum++) {
    const page = await pdf.getPage(pageNum);
    const viewport = page.getViewport({ scale: 1.4 });

    const pageWrap = document.createElement("div");
    pageWrap.className = "page";

    const canvas = document.createElement("canvas");
    canvas.width = viewport.width;
    canvas.height = viewport.height;
    pageWrap.appendChild(canvas);

    const ctx = canvas.getContext("2d");
    await page.render({ canvasContext: ctx, viewport }).promise;

    const textLayerDiv = document.createElement("div");
    textLayerDiv.className = "textLayer";
    textLayerDiv.style.width = viewport.width + "px";
    textLayerDiv.style.height = viewport.height + "px";
    pageWrap.appendChild(textLayerDiv);

    pagesEl.appendChild(pageWrap);

    const textContent = await page.getTextContent();
    pdfjsLib.renderTextLayer({
      textContentSource: textContent,
      container: textLayerDiv,
      viewport: viewport,
      textDivs: [],
    });

    textLayerDiv.addEventListener("mouseup", () => {
      const sel = window.getSelection();
      const selectedText = sel ? sel.toString().trim() : "";
      if (selectedText.length > 0) {
        Streamlit.setComponentValue({ selectedText, pageNumber: pageNum });
      }
    });
  }

  // Update iframe height after render
  setTimeout(() => Streamlit.setFrameHeight(document.body.scrollHeight), 200);
}

function onRender(event) {
  const args = event.detail.args || {};
  const pdfB64 = args.pdf_b64;
  const height = args.height || 650;

  Streamlit.setFrameHeight(height);

  if (!pdfB64) return;
  if (pdfB64 === lastPdfB64) return;

  lastPdfB64 = pdfB64;
  renderPdf(pdfB64).catch((err) => {
    console.error("PDF render error:", err);
    Streamlit.setComponentValue({ selectedText: "", pageNumber: null, error: String(err) });
  });
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
Streamlit.setComponentReady();
Streamlit.setFrameHeight(650);
