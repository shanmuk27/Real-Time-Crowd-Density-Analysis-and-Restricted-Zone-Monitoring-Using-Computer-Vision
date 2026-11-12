/* Camera capture engine.
 *
 * Lives in the parent shell (app.html) rather than in a sub-page, because the
 * content iframe is reloaded on every navigation: a video file grabbed on the
 * camera-management page would be discarded the moment the user opens live
 * cameras. Here the File and the webcam stream are held for the whole session.
 *
 * Nothing is uploaded but single JPEG frames. The source file never leaves the
 * browser and nothing is written to disk.
 */
const Capture = (function () {
  var FPS = 5;
  var FRAME_INTERVAL_MS = Math.round(1000 / FPS);
  var FACE_MAX_EDGE = 640;
  var CROWD_MAX_EDGE = 1024;

  // cameraId -> { kind, video, url, timer, lastSeq, running }
  var sources = {};
  var canvas = document.createElement("canvas");
  var ctx = canvas.getContext("2d", { willReadFrequently: false });

  function hasFrames(video) {
    return video && video.readyState >= 2 && video.videoWidth > 0 && video.videoHeight > 0;
  }

  function captureFrame(entry) {
    var video = entry.video;
    // One in-flight upload per source prevents stale frames from piling up
    // behind CPU inference. The engine receives the next fresh frame instead.
    if (!hasFrames(video) || entry.uploading) return;

    var maxEdge = entry.analysisMode === "crowd" ? CROWD_MAX_EDGE : FACE_MAX_EDGE;
    var scale = Math.min(1, maxEdge / Math.max(video.videoWidth, video.videoHeight));
    var w = Math.max(1, Math.round(video.videoWidth * scale));
    var h = Math.max(1, Math.round(video.videoHeight * scale));

    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }

    try {
      ctx.drawImage(video, 0, 0, w, h);
    } catch (err) {
      return;
    }

    entry.uploading = true;
    canvas.toBlob(function (blob) {
      if (!blob) {
        entry.uploading = false;
        return;
      }
      entry.running = true;
      API.postFrame(entry.cameraId, blob, entry.kind).then(function (res) {
        if (!res.ok) entry.running = false;
      }).finally(function () {
        entry.uploading = false;
      });
    }, "image/jpeg", 0.7);
  }

  function startPump(entry) {
    stopPump(entry);
    entry.timer = window.setInterval(function () {
      // A webcam track can end on its own (device unplugged, permission
      // revoked). Stopping here lets the feed fall through to offline.
      if (entry.kind === "webcam" && entry.stream) {
        var live = entry.stream.getVideoTracks().some(function (t) { return t.readyState === "live"; });
        if (!live) {
          stop(entry.cameraId);
          return;
        }
      }
      captureFrame(entry);
    }, FRAME_INTERVAL_MS);
  }

  function stopPump(entry) {
    if (entry && entry.timer) {
      window.clearInterval(entry.timer);
      entry.timer = null;
    }
  }

  function releaseVideo(entry) {
    if (!entry || !entry.video) return;
    entry.video.pause();
    if (entry.kind === "webcam") {
      entry.video.srcObject = null;
    } else {
      entry.video.removeAttribute("src");
    }
    entry.video.load();
    entry.video.remove();
  }

  function stop(cameraId) {
    var entry = sources[cameraId];
    if (!entry) return;
    stopPump(entry);
    if (entry.stream) {
      entry.stream.getTracks().forEach(function (t) { t.stop(); });
    }
    if (entry.url) URL.revokeObjectURL(entry.url);
    releaseVideo(entry);
    delete sources[cameraId];
  }

  function makeVideo() {
    var video = document.createElement("video");
    video.muted = true;
    video.playsInline = true;
    video.setAttribute("muted", "");
    video.autoplay = true;
    video.style.display = "none";
    document.body.appendChild(video);
    return video;
  }

  function addVideoFile(cameraId, file, analysisMode) {
    stop(cameraId);

    var video = makeVideo();
    var url = URL.createObjectURL(file);
    var entry = {
      cameraId: cameraId,
      kind: "Video File",
      analysisMode: analysisMode || "crowd",
      video: video,
      url: url,
      stream: null,
      timer: null,
      running: false,
      uploading: false,
    };
    sources[cameraId] = entry;

    video.src = url;
    video.loop = true;
    video.preload = "auto";
    video.addEventListener("error", function () { entry.running = false; });
    var started = video.play();
    if (started && started.catch) started.catch(function () { entry.running = false; });

    startPump(entry);
  }

  function addWebcam(cameraId, analysisMode) {
    stop(cameraId);

    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      return Promise.reject(new Error("This browser cannot capture a webcam feed."));
    }

    return navigator.mediaDevices.getUserMedia({ video: true }).then(function (stream) {
      var video = makeVideo();
      var entry = {
        cameraId: cameraId,
        kind: "Web Cam",
        analysisMode: analysisMode || "crowd",
        video: video,
        url: null,
        stream: stream,
        timer: null,
        running: true,
      uploading: false,
      };
      sources[cameraId] = entry;
      video.srcObject = stream;
      var started = video.play();
      if (started && started.catch) started.catch(function () { entry.running = false; });
      startPump(entry);
    });
  }

  function add(cameraId, type, file, analysisMode) {
    if (type === "Video File") {
      if (!file) return Promise.reject(new Error("No video file provided."));
      addVideoFile(cameraId, file, analysisMode);
      return Promise.resolve({ ok: true });
    }
    if (type === "Web Cam") {
      return addWebcam(cameraId, analysisMode);
    }
    // IP cameras need a transcoding backend; not wired up yet.
    return Promise.resolve({ ok: true, ignored: true });
  }

  function handleMessage(event) {
    if (event.origin !== window.location.origin) return;
    var msg = event.data;
    if (!msg || typeof msg !== "object") return;

    if (msg.type === "camera:add-source" && msg.cameraId) {
      add(msg.cameraId, msg.cameraType, msg.file || null, msg.analysisMode || "crowd").catch(function (err) {
        if (event.source && event.source.postMessage) {
          event.source.postMessage({
            type: "camera:source-error",
            cameraId: msg.cameraId,
            message: err && err.message ? err.message : "Could not start the feed.",
          }, window.location.origin);
        }
      });
    } else if (msg.type === "camera:remove" && msg.cameraId) {
      stop(msg.cameraId);
    }
  }

  function registerHost() {
    if (Capture._registered) return;
    Capture._registered = true;
    window.addEventListener("message", handleMessage);
    window.addEventListener("pagehide", function () {
      Object.keys(sources).forEach(stop);
    });
  }

  return {
    registerHost: registerHost,
    add: add,
    stop: stop,
    // Exposes the source kind so callers (and tests) can confirm a webcam
    // really went through the getUserMedia branch rather than a file path.
    kind: function (cameraId) {
      var entry = sources[cameraId];
      return entry ? entry.kind : null;
    },
    running: function (cameraId) {
      var entry = sources[cameraId];
      return !!(entry && entry.running);
    },
  };
})();

if (typeof window !== "undefined") window.Capture = Capture;
