const tabsShell = document.getElementById("tabs-shell");
const tabsHeader = document.querySelector(".tabs-header");
const tabsActivePill = document.getElementById("tabs-active-pill");
const tabButtons = document.querySelectorAll(".tab-button");
const tabPanels = document.querySelectorAll(".tab-panel");
const activeTabInput = document.getElementById("active-tab-input");

const browseButton = document.getElementById("browse-button");
const fileInput = document.getElementById("cell-image-input");
const selectedName = document.getElementById("selected-name");
const dropzone = document.getElementById("dropzone");
const normalizeToggle = document.getElementById("normalize-toggle");
const previewCard = document.getElementById("preview-card");
const previewImage = document.getElementById("preview-image");
const previewCaption = document.getElementById("preview-caption");
const resultsCard = document.getElementById("results-card");
const classifiedPreviewWithNorm = document.getElementById("classified-preview-with-norm");
const classifiedPreviewWithoutNorm = document.getElementById("classified-preview-without-norm");

const inferenceForm = document.getElementById("inference-form");
const analyzeButton = document.getElementById("analyze-button");
let previewRequestId = 0;

function updateTabIndicator() {
  if (!tabsHeader || !tabsActivePill) {
    return;
  }

  const activeButton = tabsHeader.querySelector(".tab-button.is-active");
  if (!activeButton) {
    tabsActivePill.style.opacity = "0";
    return;
  }

  const tabIndicatorInset = 10;
  const indicatorLeft = activeButton.offsetLeft + tabIndicatorInset;
  const indicatorWidth = Math.max(activeButton.offsetWidth - tabIndicatorInset * 2, 28);
  tabsActivePill.style.width = `${indicatorWidth}px`;
  tabsActivePill.style.transform = `translateX(${indicatorLeft}px)`;
  tabsActivePill.style.opacity = "1";
}

function setActiveTab(tabName) {
  tabButtons.forEach((button) => {
    const isActive = button.dataset.tab === tabName;
    button.classList.toggle("is-active", isActive);
  });

  tabPanels.forEach((panel) => {
    const shouldShow = panel.id === `tab-${tabName}`;
    panel.classList.toggle("is-active", shouldShow);
  });

  if (activeTabInput) {
    activeTabInput.value = tabName;
  }

  updateTabIndicator();
}

tabButtons.forEach((button) => {
  button.addEventListener("click", () => {
    setActiveTab(button.dataset.tab);
  });
});

if (tabsShell && !document.querySelector(".tab-button.is-active")) {
  setActiveTab("blood-test");
}

window.addEventListener("resize", () => {
  updateTabIndicator();
});

requestAnimationFrame(() => {
  updateTabIndicator();
});

function setSelectedName(file) {
  if (!selectedName) {
    return;
  }
  selectedName.textContent = file ? file.name : "No file selected";
}

function showPreviewCard(uri, caption) {
  if (!previewCard || !previewImage || !previewCaption) {
    return;
  }
  previewCard.classList.remove("is-hidden");
  previewImage.src = uri;
  previewCaption.textContent = caption || "";
}

function hidePreviewCard() {
  if (!previewCard) {
    return;
  }
  previewCard.classList.add("is-hidden");
}

function hideResultsCard() {
  if (!resultsCard) {
    return;
  }
  resultsCard.classList.add("is-hidden");
}

function getClassifiedPreviewUri(normalizeEnabled) {
  if (normalizeEnabled) {
    return classifiedPreviewWithNorm ? classifiedPreviewWithNorm.value : "";
  }
  return classifiedPreviewWithoutNorm ? classifiedPreviewWithoutNorm.value : "";
}

function clearClassificationDisplay() {
  hideResultsCard();
}

async function refreshModelInputPreview() {
  if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
    return;
  }

  const requestId = ++previewRequestId;
  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("cell_image", file, file.name);
  formData.append("normalize", normalizeToggle && normalizeToggle.checked ? "on" : "");

  try {
    const response = await fetch("/preview-model-input", {
      method: "POST",
      body: formData,
    });
    if (requestId !== previewRequestId) {
      return;
    }
    const payload = await response.json();
    if (!response.ok || !payload.preview_uri) {
      hidePreviewCard();
      return;
    }
    showPreviewCard(payload.preview_uri, payload.preview_caption || "");
    setSelectedName(file);
  } catch (_error) {
    if (requestId !== previewRequestId) {
      return;
    }
    hidePreviewCard();
  }
}

if (browseButton && fileInput) {
  browseButton.addEventListener("click", () => {
    fileInput.click();
  });
}

if (fileInput) {
  fileInput.addEventListener("change", async () => {
    const file = fileInput.files && fileInput.files.length > 0 ? fileInput.files[0] : null;
    if (file) {
      clearClassificationDisplay();
      if (analyzeButton) {
        analyzeButton.disabled = false;
      }
    }
    setSelectedName(file);
    await refreshModelInputPreview();
  });
}

if (normalizeToggle) {
  normalizeToggle.addEventListener("change", async () => {
    if (!fileInput || !fileInput.files || fileInput.files.length === 0) {
      const fallbackUri = getClassifiedPreviewUri(normalizeToggle.checked);
      if (fallbackUri) {
        showPreviewCard(fallbackUri, normalizeToggle.checked
          ? "Model input preview: center-cropped 224x224 with LAB normalization."
          : "Model input preview: center-cropped 224x224 without LAB normalization.");
      }
      return;
    }
    await refreshModelInputPreview();
  });
}

if (dropzone && fileInput) {
  const dropEvents = ["dragenter", "dragover", "dragleave", "drop"];
  dropEvents.forEach((eventName) => {
    dropzone.addEventListener(eventName, (event) => {
      event.preventDefault();
      event.stopPropagation();
    });
  });

  dropzone.addEventListener("dragenter", () => {
    dropzone.classList.add("is-drag");
  });
  dropzone.addEventListener("dragover", () => {
    dropzone.classList.add("is-drag");
  });
  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("is-drag");
  });
  dropzone.addEventListener("drop", async (event) => {
    dropzone.classList.remove("is-drag");
    const files = event.dataTransfer && event.dataTransfer.files ? event.dataTransfer.files : null;
    if (!files || files.length === 0) {
      return;
    }

    const transfer = new DataTransfer();
    transfer.items.add(files[0]);
    fileInput.files = transfer.files;
    clearClassificationDisplay();
    if (analyzeButton) {
      analyzeButton.disabled = false;
    }
    setSelectedName(files[0]);
    await refreshModelInputPreview();
  });
}

if (inferenceForm && analyzeButton) {
  inferenceForm.addEventListener("submit", () => {
    clearClassificationDisplay();
    inferenceForm.classList.add("is-loading");
    analyzeButton.disabled = true;
  });
}

const cellImages = document.querySelectorAll(".cell-image");
cellImages.forEach((image) => {
  const imageWrapper = image.closest(".image-wrap");
  if (!imageWrapper) {
    return;
  }

  const handleImageLoaded = () => {
    image.classList.remove("loading");
    imageWrapper.classList.add("loaded");
  };

  if (image.complete) {
    handleImageLoaded();
  } else {
    image.addEventListener("load", handleImageLoaded, { once: true });
    image.addEventListener("error", handleImageLoaded, { once: true });
  }
});

const currentYear = document.getElementById("current-year");
if (currentYear) {
  currentYear.textContent = String(new Date().getFullYear());
}
