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

  async removeCamera(cameraId) {
    return this.post("/api/cameras/" + cameraId + "/remove/", {});
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

  async getNotifications() {
    const res = await this.get("/api/notifications/");
    return res.notifications || [];
  },

  async getTeamMembers() {
    const res = await this.get("/api/team/");
    return res.members || [];
  },
};
