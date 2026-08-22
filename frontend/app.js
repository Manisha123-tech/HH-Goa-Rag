/* ============================================================
   HH Goa 2026 — Voice-Enabled RAG
   Frontend <-> FastAPI Connection
   ============================================================ */


/* ============================================================
   1. CONFIG
   ============================================================ */

const CONFIG = {

  API_BASE_URL: "http://localhost:8000",

  ENDPOINTS: {
    ASK: "/ask",
    VOICE: "/voice-query",
    HEALTH: "/health"
  },

  HEALTH_CHECK_TIMEOUT: 3000

};


/* ============================================================
   2. DOM REFERENCES
   ============================================================ */

const el = (id) => document.getElementById(id);

const dom = {

  themeToggle: el("themeToggle"),
  themeIconDark: el("themeIconDark"),
  themeIconLight: el("themeIconLight"),

  healthDot: el("healthDot"),
  healthLabel: el("healthLabel"),

  pipelineTrack: el("pipelineTrack"),

  micBtn: el("micBtn"),
  micStatus: el("micStatus"),
  micHint: el("micHint"),

  micIcon: el("micIcon"),
  stopIcon: el("stopIcon"),
  spinnerIcon: el("spinnerIcon"),
  checkIcon: el("checkIcon"),

  textForm: el("textForm"),
  textInput: el("textInput"),
  askBtn: el("askBtn"),

  sampleQueries: el("sampleQueries"),

  resultsSection: el("resultsSection"),

  queryCard: el("queryCard"),
  queryText: el("queryText"),

  skeletonCard: el("skeletonCard"),

  answerCard: el("answerCard"),
  answerText: el("answerText"),
  confidenceBadge: el("confidenceBadge"),

  blockedCard: el("blockedCard"),
  blockedMessage: el("blockedMessage"),
  blockedRetryBtn: el("blockedRetryBtn"),

  errorCard: el("errorCard"),
  errorMessage: el("errorMessage"),
  errorRetryBtn: el("errorRetryBtn"),

  sourcesCard: el("sourcesCard"),
  sourceCount: el("sourceCount"),
  sourcesList: el("sourcesList"),

  latencyCard: el("latencyCard"),

  latRetrieval: el("latRetrieval"),
  latRerank: el("latRerank"),
  latGeneration: el("latGeneration"),
  latTotal: el("latTotal"),

  segRetrieval: el("segRetrieval"),
  segRerank: el("segRerank"),
  segGeneration: el("segGeneration"),

  resetBtn: el("resetBtn"),
  toast: el("toast")

};


/* ============================================================
   3. PIPELINE STAGES
   ============================================================ */

const PIPELINE_STAGES = [
  "voice",
  "transcribe",
  "retrieve",
  "generate",
  "answer"
];


/* ============================================================
   4. API FUNCTIONS
   ============================================================ */


/* ---------------- TEXT QUERY ---------------- */

async function fetchAsk(query) {

  const response = await fetch(

    CONFIG.API_BASE_URL +
    CONFIG.ENDPOINTS.ASK,

    {

      method: "POST",

      headers: {
        "Content-Type": "application/json"
      },

      body: JSON.stringify({
        query: query
      })

    }

  );


  if (!response.ok) {

    throw new Error(
      `Server responded with ${response.status}`
    );

  }


  return await response.json();

}


/* ---------------- VOICE QUERY ---------------- */

async function fetchVoice(audioBlob) {

  const formData = new FormData();


  formData.append(

    "audio",

    audioBlob,

    "voice-query.webm"

  );


  const response = await fetch(

    CONFIG.API_BASE_URL +
    CONFIG.ENDPOINTS.VOICE,

    {

      method: "POST",

      body: formData

    }

  );


  if (!response.ok) {

    throw new Error(
      `Server responded with ${response.status}`
    );

  }


  return await response.json();

}


/* ---------------- HEALTH CHECK ---------------- */

async function checkHealth() {

  try {

    const controller =
      new AbortController();


    const timeout = setTimeout(

      () => controller.abort(),

      CONFIG.HEALTH_CHECK_TIMEOUT

    );


    const response = await fetch(

      CONFIG.API_BASE_URL +
      CONFIG.ENDPOINTS.HEALTH,

      {

        signal: controller.signal

      }

    );


    clearTimeout(timeout);


    return response.ok;

  }

  catch (error) {

    return false;

  }

}


/* ============================================================
   5. APPLICATION STATE
   ============================================================ */

const STATE = {

  IDLE: "idle",

  LISTENING: "listening",

  PROCESSING: "processing",

  READY: "ready",

  ERROR: "error"

};


let currentState =
  STATE.IDLE;


let mediaRecorder =
  null;


let audioChunks =
  [];


/* ============================================================
   6. MICROPHONE STATE
   ============================================================ */

function setMicState(state) {

  currentState = state;


  dom.micBtn.classList.remove(

    "listening",

    "processing",

    "ready",

    "disabled"

  );


  [

    dom.micIcon,

    dom.stopIcon,

    dom.spinnerIcon,

    dom.checkIcon

  ].forEach(

    (icon) => {

      icon.style.display = "none";

    }

  );


  /* IDLE */

  if (state === STATE.IDLE) {

    dom.micIcon.style.display =
      "block";


    dom.micStatus.textContent =
      "Start Speaking";


    dom.micHint.textContent =
      "Tap the mic and ask a question from the knowledge base";


    dom.micBtn.setAttribute(

      "aria-label",

      "Start speaking"

    );

  }


  /* LISTENING */

  else if (
    state === STATE.LISTENING
  ) {

    dom.micBtn.classList.add(
      "listening"
    );


    dom.stopIcon.style.display =
      "block";


    dom.micStatus.textContent =
      "Listening…";


    dom.micHint.textContent =
      "Speak now — tap again to stop";


    dom.micBtn.setAttribute(

      "aria-label",

      "Stop recording"

    );

  }


  /* PROCESSING */

  else if (
    state === STATE.PROCESSING
  ) {

    dom.micBtn.classList.add(

      "processing",

      "disabled"

    );


    dom.spinnerIcon.style.display =
      "block";


    dom.micStatus.textContent =
      "Processing…";


    dom.micHint.textContent =
      "Processing your question and generating an answer";

  }


  /* READY */

  else if (
    state === STATE.READY
  ) {

    dom.micBtn.classList.add(
      "ready"
    );


    dom.checkIcon.style.display =
      "block";


    dom.micStatus.textContent =
      "Answer Ready";


    dom.micHint.textContent =
      "Tap the mic to ask another question";

  }


  /* ERROR */

  else if (
    state === STATE.ERROR
  ) {

    dom.micIcon.style.display =
      "block";


    dom.micStatus.textContent =
      "Start Speaking";


    dom.micHint.textContent =
      "Something went wrong — tap to try again";

  }

}


/* ============================================================
   7. PIPELINE FUNCTIONS
   ============================================================ */

function setPipelineStage(activeStage) {

  const activeIndex =
    PIPELINE_STAGES.indexOf(
      activeStage
    );


  const stages =
    dom.pipelineTrack.querySelectorAll(
      ".pipeline-stage"
    );


  const connectors =
    dom.pipelineTrack.querySelectorAll(
      ".pipeline-connector"
    );


  stages.forEach(

    (stage, index) => {

      stage.classList.remove(

        "active",

        "done"

      );


      if (
        index < activeIndex
      ) {

        stage.classList.add(
          "done"
        );

      }

      else if (
        index === activeIndex
      ) {

        stage.classList.add(
          "active"
        );

      }

    }

  );


  connectors.forEach(

    (connector, index) => {

      connector.classList.toggle(

        "done",

        index < activeIndex

      );

    }

  );

}


function completePipeline() {

  const stages =
    dom.pipelineTrack.querySelectorAll(
      ".pipeline-stage"
    );


  const connectors =
    dom.pipelineTrack.querySelectorAll(
      ".pipeline-connector"
    );


  stages.forEach(

    (stage) => {

      stage.classList.remove(
        "active"
      );

      stage.classList.add(
        "done"
      );

    }

  );


  connectors.forEach(

    (connector) => {

      connector.classList.add(
        "done"
      );

    }

  );

}


function resetPipeline() {

  dom.pipelineTrack
    .querySelectorAll(
      ".pipeline-stage"
    )
    .forEach(

      (stage) => {

        stage.classList.remove(

          "active",

          "done"

        );

      }

    );


  dom.pipelineTrack
    .querySelectorAll(
      ".pipeline-connector"
    )
    .forEach(

      (connector) => {

        connector.classList.remove(
          "done"
        );

      }

    );

}


/* ============================================================
   8. TOAST
   ============================================================ */

function showToast(message) {

  dom.toast.textContent =
    message;


  dom.toast.classList.add(
    "show"
  );


  clearTimeout(
    showToast.timeout
  );


  showToast.timeout =
    setTimeout(

      () => {

        dom.toast.classList.remove(
          "show"
        );

      },

      2600

    );

}


/* ============================================================
   9. RESULT HELPERS
   ============================================================ */

function hideAllResultCards() {

  [

    dom.skeletonCard,

    dom.answerCard,

    dom.blockedCard,

    dom.errorCard,

    dom.sourcesCard,

    dom.latencyCard

  ].forEach(

    (card) => {

      card.hidden = true;

    }

  );

}


function scoreClass(score) {

  if (score >= 0.65) {

    return "score-high";

  }


  if (score >= 0.45) {

    return "score-mid";

  }


  return "score-low";

}


/* ============================================================
   10. RENDER SOURCES
   ============================================================ */

function renderSources(sources) {

  dom.sourcesList.innerHTML =
    "";


  sources.forEach(

    (source, index) => {

      const score =
        Number(source.score || 0);


      const percentage =
        Math.round(score * 100);


      const item =
        document.createElement("div");


      item.className =
        "source-item";


      item.innerHTML = `

        <button
          class="source-head"
          type="button"
          aria-expanded="false"
        >

          <span class="source-idx">
            #${index + 1}
          </span>


          <span class="source-head-main">

            <p class="source-id">
              ${source.id || `Source ${index + 1}`}
            </p>

            <p class="source-preview">
              ${source.text || "No preview available"}
            </p>

          </span>


          <span class="source-score-wrap">

            <span
              class="source-score-num ${scoreClass(score)}"
            >
              ${percentage}% relevant
            </span>

            <span class="source-score-bar">

              <span
                class="source-score-fill"
                style="width:${percentage}%"
              ></span>

            </span>

          </span>


          <svg
            class="source-chevron"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
          >

            <path
              d="M6 9l6 6 6-6"
              stroke="currentColor"
              stroke-width="1.8"
              stroke-linecap="round"
              stroke-linejoin="round"
            />

          </svg>

        </button>


        <div class="source-body">

          <p>
            ${source.text || ""}
          </p>

        </div>

      `;


      item
        .querySelector(".source-head")
        .addEventListener(

          "click",

          () => {

            const isOpen =
              item.classList.toggle(
                "open"
              );


            item
              .querySelector(".source-head")
              .setAttribute(

                "aria-expanded",

                String(isOpen)

              );

          }

        );


      dom.sourcesList.appendChild(
        item
      );

    }

  );


  dom.sourceCount.textContent =
    `${sources.length} chunk${
      sources.length === 1
        ? ""
        : "s"
    } retrieved`;

}


/* ============================================================
   11. RENDER LATENCY
   ============================================================ */

function renderLatency(latency) {

  const retrieval =
    Number(
      latency.retrieval_ms || 0
    );


  /*
     Your current backend does not return reranking_ms.
     So this safely displays 0.00 ms.
  */

  const reranking =
    Number(
      latency.reranking_ms || 0
    );


  const generation =
    Number(
      latency.generation_ms || 0
    );


  const total =
    Number(
      latency.total_ms || 0
    );


  dom.latRetrieval.textContent =
    `${retrieval.toFixed(2)} ms`;


  dom.latRerank.textContent =
    `${reranking.toFixed(2)} ms`;


  dom.latGeneration.textContent =
    `${generation.toFixed(2)} ms`;


  dom.latTotal.textContent =
    `${total.toFixed(2)} ms`;


  const safeTotal =
    Math.max(total, 1);


  requestAnimationFrame(

    () => {

      dom.segRetrieval.style.width =
        `${(retrieval / safeTotal) * 100}%`;


      dom.segRerank.style.width =
        `${(reranking / safeTotal) * 100}%`;


      dom.segGeneration.style.width =
        `${(generation / safeTotal) * 100}%`;

    }

  );

}


/* ============================================================
   12. RENDER RESULT
   ============================================================ */

function renderResult(result) {

  hideAllResultCards();


  /* BLOCKED */

  if (

    result.blocked ||

    (
      result.success === false &&
      result.error
    )

  ) {

    dom.blockedMessage.textContent =

      result.error ||

      "The knowledge base does not contain enough relevant information to answer this question.";


    dom.blockedCard.hidden =
      false;

  }


  /* SUCCESS */

  else if (
    result.success
  ) {

    dom.answerText.textContent =

      result.answer ||

      "No answer was generated.";


    /*
       Confidence is optional because
       current main.py does not return it.
    */

    if (

      result.confidence !== undefined &&

      result.confidence !== null

    ) {

      dom.confidenceBadge.textContent =

        `${Math.round(
          result.confidence * 100
        )}% confidence`;

    }

    else {

      dom.confidenceBadge.textContent =
        "";

    }


    dom.answerCard.hidden =
      false;

  }


  /* SOURCES */

  if (

    result.sources &&

    result.sources.length > 0

  ) {

    renderSources(
      result.sources
    );


    dom.sourcesCard.hidden =
      false;

  }


  /* LATENCY */

  if (
    result.latency
  ) {

    renderLatency(
      result.latency
    );


    dom.latencyCard.hidden =
      false;

  }

}


/* ============================================================
   13. RENDER ERROR
   ============================================================ */

function renderError(message) {

  hideAllResultCards();


  dom.errorMessage.textContent =

    message ||

    "Unable to connect to the RAG server. Check that the backend is running.";


  dom.errorCard.hidden =
    false;

}


/* ============================================================
   14. RUN QUERY
   ============================================================ */

async function runQuery(

  query,

  {

    viaVoice = false,

    audioBlob = null

  } = {}

) {

  if (
    !query &&
    !audioBlob
  ) {

    return;

  }


  dom.resultsSection.hidden =
    false;


  dom.queryText.textContent =

    query ||

    "Transcribing your voice…";


  dom.queryCard.hidden =
    false;


  hideAllResultCards();


  dom.skeletonCard.hidden =
    false;


  dom.askBtn.disabled =
    true;


  setMicState(
    STATE.PROCESSING
  );


  /* TEXT */

  if (!viaVoice) {

    setPipelineStage(
      "retrieve"
    );

  }


  /* VOICE */

  else {

    setPipelineStage(
      "transcribe"
    );

  }


  try {

    let result;


    if (viaVoice) {

      result =
        await fetchVoice(
          audioBlob
        );


      /*
         Voice request is complete.
         Now visually show RAG stages.
      */

      setPipelineStage(
        "retrieve"
      );

      await new Promise(
        resolve =>
          setTimeout(resolve, 250)
      );


      setPipelineStage(
        "generate"
      );

      await new Promise(
        resolve =>
          setTimeout(resolve, 250)
      );

    }

    else {

      result =
        await fetchAsk(
          query
        );


      setPipelineStage(
        "generate"
      );

    }


    /*
       Update query with backend response.
       For voice this shows transcription.
    */

    dom.queryText.textContent =

      result.query ||

      result.transcript ||

      query;


    setPipelineStage(
      "answer"
    );


    completePipeline();


    renderResult(
      result
    );


    setMicState(
      STATE.READY
    );

  }


  catch (error) {

    console.error(error);


    renderError(

      "Unable to connect to the RAG server. Make sure FastAPI is running on http://localhost:8000."

    );


    setMicState(
      STATE.ERROR
    );


    showToast(
      "Request failed"
    );

  }


  finally {

    dom.skeletonCard.hidden =
      true;


    dom.askBtn.disabled =
      false;

  }

}


/* ============================================================
   15. RESET
   ============================================================ */

function resetToIdle() {

  dom.resultsSection.hidden =
    true;


  hideAllResultCards();


  dom.queryCard.hidden =
    true;


  dom.textInput.value =
    "";


  resetPipeline();


  setMicState(
    STATE.IDLE
  );

}


/* ============================================================
   16. VOICE RECORDING
   ============================================================ */

async function toggleRecording() {

  if (
    currentState === STATE.PROCESSING
  ) {

    return;

  }


  /* STOP RECORDING */

  if (
    currentState === STATE.LISTENING
  ) {

    if (

      mediaRecorder &&

      mediaRecorder.state !== "inactive"

    ) {

      mediaRecorder.stop();

    }


    return;

  }


  /* CHECK MICROPHONE SUPPORT */

  if (

    !navigator.mediaDevices ||

    !navigator.mediaDevices.getUserMedia

  ) {

    showToast(
      "Microphone is not supported in this browser."
    );


    return;

  }


  try {

    const stream =
      await navigator
        .mediaDevices
        .getUserMedia({

          audio: true

        });


    audioChunks =
      [];


    mediaRecorder =
      new MediaRecorder(
        stream
      );


    mediaRecorder.ondataavailable =

      (event) => {

        if (
          event.data.size > 0
        ) {

          audioChunks.push(
            event.data
          );

        }

      };


    mediaRecorder.onstop =

      async () => {

        stream
          .getTracks()
          .forEach(

            track => track.stop()

          );


        const audioBlob =
          new Blob(

            audioChunks,

            {
              type: "audio/webm"
            }

          );


        await runQuery(

          null,

          {

            viaVoice: true,

            audioBlob: audioBlob

          }

        );

      };


    mediaRecorder.start();


    setMicState(
      STATE.LISTENING
    );


    setPipelineStage(
      "voice"
    );

  }


  catch (error) {

    console.error(error);


    showToast(

      "Microphone access was denied. Please allow microphone access and try again."

    );


    setMicState(
      STATE.ERROR
    );

  }

}


/* ============================================================
   17. THEME
   ============================================================ */

function applyTheme(theme) {

  document.body.setAttribute(

    "data-theme",

    theme

  );


  dom.themeIconDark.style.display =

    theme === "dark"

      ? "block"

      : "none";


  dom.themeIconLight.style.display =

    theme === "light"

      ? "block"

      : "none";


  try {

    localStorage.setItem(

      "hhgoa-theme",

      theme

    );

  }

  catch (error) {

    console.warn(
      "Theme could not be saved."
    );

  }

}


function initTheme() {

  let theme =
    "dark";


  try {

    theme =

      localStorage.getItem(
        "hhgoa-theme"
      )

      ||

      "dark";

  }

  catch (error) {

    theme =
      "dark";

  }


  applyTheme(
    theme
  );

}


/* ============================================================
   18. EVENTS
   ============================================================ */

dom.micBtn.addEventListener(

  "click",

  toggleRecording

);


dom.textForm.addEventListener(

  "submit",

  (event) => {

    event.preventDefault();


    const query =
      dom.textInput.value.trim();


    if (!query) {

      return;

    }


    runQuery(
      query
    );

  }

);


dom.sampleQueries.addEventListener(

  "click",

  (event) => {

    const chip =
      event.target.closest(
        ".chip"
      );


    if (!chip) {

      return;

    }


    const query =
      chip.dataset.q;


    dom.textInput.value =
      query;


    runQuery(
      query
    );

  }

);


dom.resetBtn.addEventListener(

  "click",

  resetToIdle

);


dom.blockedRetryBtn.addEventListener(

  "click",

  resetToIdle

);


dom.errorRetryBtn.addEventListener(

  "click",

  () => {

    const query =
      dom.queryText.textContent.trim();


    resetToIdle();


    if (query) {

      dom.textInput.value =
        query;


      runQuery(
        query
      );

    }

  }

);


dom.themeToggle.addEventListener(

  "click",

  () => {

    const currentTheme =

      document.body.getAttribute(
        "data-theme"
      );


    const nextTheme =

      currentTheme === "dark"

        ? "light"

        : "dark";


    applyTheme(
      nextTheme
    );

  }

);


/* ============================================================
   19. INITIALIZATION
   ============================================================ */

(async function init() {

  initTheme();


  const isBackendOnline =
    await checkHealth();


  dom.healthDot.classList.remove(

    "ok",

    "down"

  );


  if (isBackendOnline) {

    dom.healthDot.classList.add(
      "ok"
    );


    dom.healthLabel.textContent =
      "Backend connected";

  }

  else {

    dom.healthDot.classList.add(
      "down"
    );


    dom.healthLabel.textContent =
      "Backend offline";

  }

})();