const API = {
  getCookie(name) {
    const value = `; ${document.cookie}`;
    const parts = value.split(`; ${name}=`);
    if (parts.length === 2) return parts.pop().split(";").shift();
    return null;
  },

  async get(path) {
    try {
      const res = await fetch(path);
      return await res.json();
    } catch {
      return {};
    }
  },

  async post(path, data) {
    try {
      const res = await fetch(path, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-CSRFToken": this.getCookie("csrftoken"),
        },
        body: JSON.stringify(data),
      });
      return await res.json();
    } catch {
      return { ok: false, message: "Could not reach the server." };
    }
  },

  async getCameras() {
    const res = await this.get("/api/cameras/");
    return res.cameras || [];
  },

  async addCamera(data) {
    return this.post("/api/cameras/add/", data);
  },

  async updateCamera(cameraId, data) {
    return this.post("/api/cameras/" + encodeURIComponent(cameraId) + "/update/", data);
  },

  async addFaceEnrollments(cameraId, name, files) {
    const form = new FormData();
    form.append("name", name);
    Array.from(files || []).forEach(function (file) { form.append("faces", file); });
    try {
      const res = await fetch("/api/cameras/" + encodeURIComponent(cameraId) + "/enrollments/", {
        method: "POST",
        headers: { "X-CSRFToken": this.getCookie("csrftoken") },
        body: form,
      });
      return await res.json();
    } catch {
      return { ok: false, message: "Could not enroll the supplied faces." };
    }
  },

  async detectFaceCandidates(cameraId, files) {
    const form = new FormData();
    Array.from(files || []).forEach(function (file) { form.append("faces", file); });
    try {
      const res = await fetch("/api/cameras/" + encodeURIComponent(cameraId) + "/face-candidates/", {
        method: "POST", headers: { "X-CSRFToken": this.getCookie("csrftoken") }, body: form,
      });
      return await res.json();
    } catch {
      return { ok: false, message: "Could not detect faces in the selected folder." };
    }
  },

  async confirmFaceCandidates(cameraId, candidates) {
    return this.post("/api/cameras/" + encodeURIComponent(cameraId) + "/face-candidates/confirm/", { candidates: candidates });
  },

  async removeCamera(cameraId) {
    return this.post("/api/cameras/" + encodeURIComponent(cameraId) + "/remove/", {});
  },

  async postFrame(cameraId, blob, kind) {
    const url = "/api/cameras/" + encodeURIComponent(cameraId) + "/frame/" +
      (kind ? "?kind=" + encodeURIComponent(kind) : "");
    try {
      const res = await fetch(url, {
        method: "POST",
        headers: {
          "Content-Type": "image/jpeg",
          "X-CSRFToken": this.getCookie("csrftoken"),
        },
        body: blob,
      });
      return await res.json();
    } catch {
      return { ok: false, message: "Could not reach the server." };
    }
  },

  async getCameraFeedStatus() {
    const res = await this.get("/api/camera-feeds/status/");
    return res.feeds || {};
  },

  async getEngineStatus() {
    return this.get("/api/engines/status/");
  },

  async getDashboardMetrics() {
    return this.get("/api/dashboard-metrics/");
  },

  async fetchFrame(cameraId) {
    // Returned as an object URL rather than handed straight to an <img> src:
    // reassigning src cancels the load still in flight, which leaves the
    // element at naturalWidth 0 when frames are polled faster than they decode.
    try {
      const res = await fetch(
        "/api/cameras/" + encodeURIComponent(cameraId) + "/frame/?t=" + Date.now(),
        { cache: "no-store", credentials: "same-origin" }
      );
      if (!res.ok) return null;
      const blob = await res.blob();
      return URL.createObjectURL(blob);
    } catch {
      return null;
    }
  },

  async getLiveCrowd() {
    return this.get("/api/live-crowd/");
  },

  async addCrowdRecord(cameraId, count) {
    return this.post("/api/crowd-record/add/", { camera_id: cameraId, count: count });
  },

  async getMonthlyCrowd(month, year) {
    const params = new URLSearchParams();
    if (month) params.set("month", month);
    if (year) params.set("year", year);
    const res = await this.get("/api/monthly-crowd/?" + params.toString());
    return res.monthly_crowd || [];
  },

  async getPersons() {
    const res = await this.get("/api/persons/");
    return res.persons || [];
  },

  async removeUnverifiedPerson(personId) {
    return this.post("/api/persons/" + encodeURIComponent(personId) + "/remove/", {});
  },

  async getNotifications() {
    const res = await this.get("/api/notifications/");
    return res.notifications || [];
  },

  async getTeamMembers() {
    const res = await this.get("/api/team/");
    return res.members || [];
  },
};
